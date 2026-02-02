# src/eun/attendance/routes.py
import os
from datetime import date, timedelta

import pandas as pd
import requests
from flask import Blueprint, request, jsonify

bp = Blueprint("eun_attendance", __name__, url_prefix="/eun/attendance")

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


def _to_df(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # workDate
    if "workDate" in df.columns:
        df["workDate"] = pd.to_datetime(df["workDate"], errors="coerce").dt.date
    else:
        df["workDate"] = pd.NaT

    # checkIn/checkOut
    df["checkInDt"] = pd.to_datetime(df["checkIn"], errors="coerce") if "checkIn" in df.columns else pd.NaT
    df["checkOutDt"] = pd.to_datetime(df["checkOut"], errors="coerce") if "checkOut" in df.columns else pd.NaT

    # minutes cols
    for col in ["normalWorkMinutes", "overtimeWorkMinutes", "unpaidMinutes", "totalWorkMinutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0

    # 필수키 방어
    if "empId" not in df.columns:
        df["empId"] = None
    if "empName" not in df.columns:
        df["empName"] = None
    if "workStatus" not in df.columns:
        df["workStatus"] = None
    if "workType" not in df.columns:
        df["workType"] = None

    df = df[df["empId"].notna()]
    df = df[df["workDate"].notna()]

    return df


def _date_str(d: date) -> str:
    return d.isoformat()


def _default_range_30days():
    end = date.today()
    start = end - timedelta(days=29)
    return start, end


def _build_alerts(df: pd.DataFrame) -> list[dict]:
    """
    이상 감지(룰 기반):
    - ABSENT 존재
    - NIGHT인데 checkOut 없음
    - LATE 3회 이상
    - unpaidMinutes 합이 큰 경우(예: 180분 이상)
    - overtimeMinutes 합이 큰 경우(예: 600분 이상)
    """
    if df.empty:
        return []

    alerts = []
    by_emp = df.groupby(["empId", "empName"], dropna=False)

    for (emp_id, emp_name), g in by_emp:
        late_cnt = (g["workStatus"] == "LATE").sum()
        absent_cnt = (g["workStatus"] == "ABSENT").sum()

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
                "empName": emp_name or emp_id,
                "score": score,
                "reason": ", ".join(reasons)
            })

    alerts.sort(key=lambda x: x["score"], reverse=True)
    return alerts[:10]


def _build_employee_report(df: pd.DataFrame, emp_id: str) -> dict:
    g = df[df["empId"] == emp_id].copy()
    if g.empty:
        return {"empId": emp_id, "message": "해당 기간 데이터가 없습니다."}

    emp_name = g["empName"].dropna().iloc[0] if g["empName"].notna().any() else emp_id

    status_counts = g["workStatus"].fillna("UNKNOWN").value_counts().to_dict()

    g = g.sort_values("workDate")
    labels = [d.isoformat() for d in g["workDate"]]

    series = {
        "labels": labels,
        "normal": g["normalWorkMinutes"].tolist(),
        "overtime": g["overtimeWorkMinutes"].tolist(),
        "unpaid": g["unpaidMinutes"].tolist()
    }

    late_cnt = int((g["workStatus"] == "LATE").sum())
    early_cnt = int((g["workStatus"] == "EARLY_LEAVE").sum())
    absent_cnt = int((g["workStatus"] == "ABSENT").sum())
    pending_cnt = int((g["workStatus"] == "PENDING").sum())

    unpaid_sum = int(g["unpaidMinutes"].sum())
    overtime_sum = int(g["overtimeWorkMinutes"].sum())

    night_cnt = int((g["workType"] == "NIGHT").sum())
    night_no_checkout = int(((g["workType"] == "NIGHT") & (g["checkOutDt"].isna())).sum())

    bullets = [
        f"최근 기간 내 지각 {late_cnt}회, 조퇴 {early_cnt}회, 결근 {absent_cnt}회 입니다."
    ]
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

    actions = []
    if late_cnt >= 3:
        actions.append("출근 알림/지각 사유 확인")
    if night_no_checkout > 0:
        actions.append("퇴근 미기록 사유 확인 및 관리자 수정 권장")
    if absent_cnt > 0:
        actions.append("결근 처리/증빙 확인")

    return {
        "empId": emp_id,
        "empName": emp_name,
        "statusCounts": status_counts,
        "series": series,
        "analysis": {
            "summary": summary,
            "bullets": bullets,
            "recommendedActions": actions if actions else ["현 상태 유지 및 정기 모니터링"]
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

    data, code, err = _spring_get("/admin/attendance/list", {
        "startDate": start,
        "endDate": end
    })
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
        "empId": emp_id
    })
    if code != 200:
        return jsonify({"message": "Spring 호출 실패", "detail": err}), code

    df = _to_df(data)
    report = _build_employee_report(df, emp_id)
    report["startDate"] = start
    report["endDate"] = end
    return jsonify(report)
