import os
from datetime import date, datetime, timedelta

import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

SPRING_BASE = os.getenv("SPRING_BASE", "http://localhost:8080")

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173"])


def _spring_get(path: str, params: dict):
    """
    React -> Flask 요청에 포함된 세션 쿠키(JSESSIONID 등)를 그대로 Spring으로 전달.
    """
    url = f"{SPRING_BASE}{path}"
    cookies = request.cookies  # Flask가 받은 쿠키(브라우저 세션)
    headers = {}
    # 필요하면 Authorization 헤더도 전달 가능
    # if request.headers.get("Authorization"):
    #     headers["Authorization"] = request.headers.get("Authorization")

    r = requests.get(url, params=params, cookies=cookies, headers=headers, timeout=10)
    if r.status_code != 200:
        return None, r.status_code, r.text
    return r.json(), 200, None


def _to_df(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # 필드명은 AttendanceListResponseDto에 맞춰야 함
    # 여기서는 너가 AdminAttendance.jsx에서 쓰는 키들을 기준으로 가정:
    # empId, empName, workDate, checkIn, checkOut, workStatus, workType, totalWorkMinutes, normalWorkMinutes, overtimeWorkMinutes, unpaidMinutes

    df["workDate"] = pd.to_datetime(df["workDate"], errors="coerce").dt.date

    # checkIn/checkOut이 문자열이면 파싱
    if "checkIn" in df.columns:
        df["checkInDt"] = pd.to_datetime(df["checkIn"], errors="coerce")
    else:
        df["checkInDt"] = pd.NaT

    if "checkOut" in df.columns:
        df["checkOutDt"] = pd.to_datetime(df["checkOut"], errors="coerce")
    else:
        df["checkOutDt"] = pd.NaT

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

        unpaid_sum = g["unpaidMinutes"].sum()
        overtime_sum = g["overtimeWorkMinutes"].sum()

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

        # 스코어가 일정 이상이면 알림
        if score >= 3 and reasons:
            alerts.append({
                "empId": emp_id,
                "empName": emp_name,
                "score": score,
                "reason": ", ".join(reasons)
            })

    alerts.sort(key=lambda x: x["score"], reverse=True)
    return alerts[:10]


def _build_employee_report(df: pd.DataFrame, emp_id: str) -> dict:
    """
    사원 1명 보고서:
    - 상태 분포(Bar)
    - 일자별 minutes(Line)
    - 텍스트 분석(템플릿)
    """
    g = df[df["empId"] == emp_id].copy()
    if g.empty:
        return {"empId": emp_id, "message": "해당 기간 데이터가 없습니다."}

    emp_name = g["empName"].dropna().iloc[0] if "empName" in g and g["empName"].notna().any() else emp_id

    # 상태 분포
    status_counts = {}
    if "workStatus" in g:
        status_counts = g["workStatus"].fillna("UNKNOWN").value_counts().to_dict()

    # 일자별 minutes
    g = g.sort_values("workDate")
    days = [d.isoformat() if pd.notna(d) else "" for d in g["workDate"]]

    series = {
        "labels": days,
        "normal": g["normalWorkMinutes"].tolist(),
        "overtime": g["overtimeWorkMinutes"].tolist(),
        "unpaid": g["unpaidMinutes"].tolist()
    }

    # 텍스트 분석(설명 가능한 규칙)
    late_cnt = (g["workStatus"] == "LATE").sum() if "workStatus" in g else 0
    early_cnt = (g["workStatus"] == "EARLY_LEAVE").sum() if "workStatus" in g else 0
    absent_cnt = (g["workStatus"] == "ABSENT").sum() if "workStatus" in g else 0
    pending_cnt = (g["workStatus"] == "PENDING").sum() if "workStatus" in g else 0

    unpaid_sum = int(g["unpaidMinutes"].sum())
    overtime_sum = int(g["overtimeWorkMinutes"].sum())

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

    # “예상 상황” 문장(AI 느낌 템플릿)
    risk = []
    if absent_cnt > 0:
        risk.append("근무 이탈(결근) 패턴")
    if late_cnt >= 3:
        risk.append("지각 증가 패턴")
    if night_no_checkout > 0:
        risk.append("퇴근 미기록으로 인한 야간근무 오인 리스크")

    if risk:
        summary = f"{emp_name} 사원은 최근 기간 내 " + ", ".join(risk) + "이(가) 관찰됩니다."
    else:
        summary = f"{emp_name} 사원은 최근 기간 내 큰 이상 패턴이 두드러지지 않습니다."

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
        "series": series,
        "analysis": {
            "summary": summary,
            "bullets": bullets,
            "recommendedActions": next_steps
        }
    }


@app.get("/attendance/anomalies")
def anomalies():
    """
    기간 내 전체 사원 이상감지 Top N
    query:
      - startDate=YYYY-MM-DD (옵션)
      - endDate=YYYY-MM-DD (옵션)
    """
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
    return jsonify({
        "startDate": start,
        "endDate": end,
        "alerts": alerts
    })


@app.get("/attendance/employee-report")
def employee_report():
    """
    사원 1명 분석 리포트
    query:
      - empId (필수)
      - startDate/endDate (옵션, 없으면 최근 30일)
    """
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
