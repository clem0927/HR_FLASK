# src/eun/attendance/routes.py
import os
from datetime import date, timedelta

import pandas as pd
import requests
from flask import Blueprint, request, jsonify

bp = Blueprint("attendance_ai", __name__, url_prefix="/attendance")

SPRING_BASE = os.getenv("SPRING_BASE", "http://localhost:8080")


def _spring_get(path: str, params: dict):
    """
    React -> Flask 요청에 포함된 세션 쿠키(JSESSIONID 등)를 그대로 Spring으로 전달.
    """
    url = f"{SPRING_BASE}{path}"
    cookies = request.cookies
    headers = {}

    r = requests.get(url, params=params, cookies=cookies, headers=headers, timeout=10)
    if r.status_code != 200:
        return None, r.status_code, r.text
    return r.json(), 200, None


def _normalize_java_time(v):
    """
    Spring/Jackson LocalDateTime이 내려오는 형태를 통일:
    - "2026-01-01T09:00:00" (str)
    - [2026,1,1,9,0,0] (list/tuple)
    - None
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None

    if isinstance(v, (list, tuple)) and len(v) >= 5:
        y, m, d, hh, mm = v[0], v[1], v[2], v[3], v[4]
        ss = v[5] if len(v) >= 6 else 0
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}T{int(hh):02d}:{int(mm):02d}:{int(ss):02d}"

    if isinstance(v, dict):
        # 혹시 dict 형태로 올 때 대비
        if "year" in v and ("monthValue" in v or "month" in v) and "dayOfMonth" in v:
            y = v.get("year")
            m = v.get("monthValue") or v.get("month")
            d = v.get("dayOfMonth")
            hh = v.get("hour", 0)
            mm = v.get("minute", 0)
            ss = v.get("second", 0)
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}T{int(hh):02d}:{int(mm):02d}:{int(ss):02d}"
        return None

    return str(v)


def _to_df(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # workDate
    df["workDate"] = pd.to_datetime(df.get("workDate"), errors="coerce").dt.date

    # checkIn/checkOut -> datetime
    if "checkIn" in df.columns:
        df["_checkInNorm"] = df["checkIn"].apply(_normalize_java_time)
        df["checkInDt"] = pd.to_datetime(df["_checkInNorm"], errors="coerce")
    else:
        df["checkInDt"] = pd.NaT

    if "checkOut" in df.columns:
        df["_checkOutNorm"] = df["checkOut"].apply(_normalize_java_time)
        df["checkOutDt"] = pd.to_datetime(df["_checkOutNorm"], errors="coerce")
    else:
        df["checkOutDt"] = pd.NaT

    # minutes columns (없으면 0)
    for col in ["normalWorkMinutes", "overtimeWorkMinutes", "unpaidMinutes", "totalWorkMinutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0

    return df


def _date_str(d: date) -> str:
    return d.isoformat()


def _default_range_30days():
    end = date.today()
    start = end - timedelta(days=29)
    return start, end


def _build_alerts(df: pd.DataFrame) -> list[dict]:
    """
    이상 감지(규칙+가벼운 스코어):
    - ABSENT 존재
    - NIGHT인데 checkOut 없음(미퇴근 위험)
    - LATE 3회 이상
    - unpaidMinutes 합이 큰 경우(예: 180분 이상)
    - overtimeMinutes 합이 큰 경우(예: 600분 이상)
    """
    if df.empty:
        return []

    alerts = []
    by_emp = df.groupby(["empId", "empName"], dropna=False)

    for (emp_id, emp_name), g in by_emp:
        g = g.copy()
        late_cnt = (g["workStatus"] == "LATE").sum() if "workStatus" in g else 0
        absent_cnt = (g["workStatus"] == "ABSENT").sum() if "workStatus" in g else 0

        night_no_checkout = 0
        if "workType" in g:
            night_no_checkout = ((g["workType"] == "NIGHT") & (g["checkOutDt"].isna())).sum()

        unpaid_sum = int(g["unpaidMinutes"].sum())
        overtime_sum = int(g["overtimeWorkMinutes"].sum())

        reasons = []
        score = 0

        if absent_cnt > 0:
            reasons.append(f"결근 {absent_cnt}회")
            score += 3 + absent_cnt

        if night_no_checkout > 0:
            reasons.append(f"NIGHT 미퇴근 {night_no_checkout}회")
            score += 3 + night_no_checkout

        if late_cnt >= 3:
            reasons.append(f"지각 {late_cnt}회")
            score += 2 + (late_cnt - 2)

        if unpaid_sum >= 180:
            reasons.append(f"무급(공제) {unpaid_sum}분")
            score += 2

        if overtime_sum >= 600:
            reasons.append(f"야근 {overtime_sum}분")
            score += 1

        if score >= 3 and reasons:
            alerts.append({
                "empId": emp_id,
                "empName": emp_name,
                "score": score,
                "reason": ", ".join(reasons)
            })

    alerts.sort(key=lambda x: x["score"], reverse=True)
    return alerts[:10]


def _minutes_from_midnight(dt: pd.Timestamp):
    if pd.isna(dt):
        return None
    return int(dt.hour * 60 + dt.minute)


def _build_series_times(g: pd.DataFrame) -> dict:
    """
    일자별 출근/퇴근 “시각” 시리즈 생성 (분 단위)
    - labels: YYYY-MM-DD
    - checkInMinutes: [0..] or None
    - checkOutMinutes: [0..] or None
    - 야간근무 등으로 퇴근이 다음날이면 +1440 해서 표현 (예: 25:10)
      -> 프론트에서 y축 max를 36*60 정도로 잡으면 선이 안 잘림
    """
    if g.empty:
        return {"labels": [], "checkInMinutes": [], "checkOutMinutes": []}

    g = g.sort_values("workDate").copy()

    labels = []
    cin = []
    cout = []

    for _, row in g.iterrows():
        wd = row.get("workDate")
        labels.append(wd.isoformat() if pd.notna(wd) else "")

        ci = row.get("checkInDt")
        co = row.get("checkOutDt")

        ci_m = _minutes_from_midnight(ci)
        co_m = _minutes_from_midnight(co)

        # ✅ 퇴근이 다음날로 넘어간 경우(야간근무 등): +1440 해서 25:xx로 표현
        #   (workDate 기준으로 "퇴근 날짜 > workDate" 인 경우)
        try:
            if pd.notna(co) and pd.notna(wd):
                if co.date() > wd:
                    co_m = (co_m if co_m is not None else None)
                    if co_m is not None:
                        co_m += 1440
        except Exception:
            pass

        cin.append(ci_m)
        cout.append(co_m)

    return {"labels": labels, "checkInMinutes": cin, "checkOutMinutes": cout}


def _build_employee_report(df: pd.DataFrame, emp_id: str) -> dict:
    g = df[df["empId"] == emp_id].copy()
    if g.empty:
        return {"empId": emp_id, "message": "해당 기간 데이터가 없습니다."}

    emp_name = (
        g["empName"].dropna().iloc[0]
        if "empName" in g and g["empName"].notna().any()
        else emp_id
    )

    # 상태 분포
    status_counts = {}
    if "workStatus" in g:
        status_counts = g["workStatus"].fillna("UNKNOWN").value_counts().to_dict()

    # ✅ 출퇴근 “시각” 시리즈
    series_times = _build_series_times(g)

    # 텍스트 분석(기존)
    late_cnt = (g["workStatus"] == "LATE").sum() if "workStatus" in g else 0
    early_cnt = (g["workStatus"] == "EARLY_LEAVE").sum() if "workStatus" in g else 0
    absent_cnt = (g["workStatus"] == "ABSENT").sum() if "workStatus" in g else 0
    pending_cnt = (g["workStatus"] == "PENDING").sum() if "workStatus" in g else 0

    unpaid_sum = int(g["unpaidMinutes"].sum()) if "unpaidMinutes" in g else 0
    overtime_sum = int(g["overtimeWorkMinutes"].sum()) if "overtimeWorkMinutes" in g else 0

    night_cnt = (g["workType"] == "NIGHT").sum() if "workType" in g else 0
    night_no_checkout = ((g["workType"] == "NIGHT") & (g["checkOutDt"].isna())).sum() if "workType" in g else 0

    bullets = []
    bullets.append(f"최근 기간 내 지각 {late_cnt}회, 조퇴 {early_cnt}회, 결근 {absent_cnt}회 입니다.")
    if unpaid_sum > 0:
        bullets.append(f"무급(공제) 시간 합계는 {unpaid_sum}분 입니다.")
    if overtime_sum > 0:
        bullets.append(f"야근 시간 합계는 {overtime_sum}분 입니다.")
    if night_cnt > 0:
        bullets.append(f"NIGHT 근무 유형이 {night_cnt}회 감지되었습니다.")
    if night_no_checkout > 0:
        bullets.append(f"⚠ NIGHT 상태에서 퇴근 미기록이 {night_no_checkout}회 있어 리스크가 있습니다.")
    if pending_cnt > 0:
        bullets.append(f"PENDING 상태가 {pending_cnt}건 남아있습니다(출근 미실행/기록 생성 직후 가능).")

    risk = []
    if absent_cnt > 0:
        risk.append("근무 이탈(결근) 패턴")
    if late_cnt >= 3:
        risk.append("지각 증가 패턴")
    if night_no_checkout > 0:
        risk.append("퇴근 미기록으로 인한 야간근무 오인 리스크")

    summary = (
        f"{emp_name} 사원은 최근 기간 내 " + ", ".join(risk) + "이(가) 관찰됩니다."
        if risk else
        f"{emp_name} 사원은 최근 기간 내 큰 이상 패턴이 두드러지지 않습니다."
    )

    action = []
    if late_cnt >= 3:
        action.append("출근 알림/지각 사유 확인")
    if night_no_checkout > 0:
        action.append("퇴근 미기록 사유 확인 및 관리자 수정 권장")
    if absent_cnt > 0:
        action.append("결근 처리/증빙 확인")

    next_steps = action if action else ["현 상태 유지 및 정기 모니터링"]

    return {
        "empId": emp_id,
        "empName": emp_name,
        "statusCounts": status_counts,
        "seriesTimes": series_times,  # ✅ 프론트가 기대하는 필드
        "analysis": {
            "summary": summary,
            "bullets": bullets,
            "recommendedActions": next_steps,
        }
    }


@bp.get("/anomalies")
def anomalies():
    start = request.args.get("startDate")
    end = request.args.get("endDate")

    if not start or not end:
        s, e = _default_range_30days()
        start = start or _date_str(s)
        end = end or _date_str(e)

    data, code, err = _spring_get("/admin/attendance/list", {"startDate": start, "endDate": end})
    if code != 200:
        return jsonify({"message": "Spring 호출 실패", "detail": err}), code

    df = _to_df(data)
    alerts = _build_alerts(df)
    return jsonify({"startDate": start, "endDate": end, "alerts": alerts})


@bp.get("/employee-report")
def employee_report():
    emp_id = request.args.get("empId")
    if not emp_id:
        return jsonify({"message": "empId는 필수입니다."}), 400

    start = request.args.get("startDate")
    end = request.args.get("endDate")

    if not start or not end:
        s, e = _default_range_30days()
        start = start or _date_str(s)
        end = end or _date_str(e)

    data, code, err = _spring_get("/admin/attendance/list", {
        "startDate": start,
        "endDate": end,
        "empId": emp_id,
    })
    if code != 200:
        return jsonify({"message": "Spring 호출 실패", "detail": err}), code

    df = _to_df(data)
    report = _build_employee_report(df, emp_id)
    report["startDate"] = start
    report["endDate"] = end
    return jsonify(report)
