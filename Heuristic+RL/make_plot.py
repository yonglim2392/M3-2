#!/usr/bin/env python
# -*- coding: utf-8 -*-

# !! 해당 코드는 개발 시 방향성 제시를 위해 개발한 코드로 실제 표출과는 다를 수 있습니다. !! #
"""
PROGRAM NAME  : make_plot.py
PROGRAMMER    : YONG
CREATION DATE : 2025-08-29

[Description]
  - CSV 형태의 생산 스케줄 데이터를 시각화하는 스크립트
  - 각 작업(PROD_NO)의 시작/종료 시간을 누적 작업 시간 좌표로 변환 후 Gantt 차트 형태로 표시
  - 작업 시간:
      - 평일 기준: WORK_START_HOUR ~ WORK_END_HOUR
      - 주말: 작업 시간 없음
  - 컬러:
      - 작업 스타일(Style)에 따라 색상 지정
      - Missed_Delivery 발생 시 텍스트 빨간색 표시
  - X축:
      - 누적 작업 시간을 기준으로 날짜 레이블 표시
      - 평일만 눈금 추가
  - Y축:
      - Assigned_Line 별로 작업 구분
  - 사용 목적:
      - 생산 라인별 작업 일정 시각화
      - 납기 지연 여부 확인 및 일정 최적화 분석
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta

# --- datetime을 누적 작업 시간 좌표로 변환하는 헬퍼 함수 ---
# 이 함수는 datetime 객체를 수치 좌표로 변환
# 각 평일은 X축에서 1단위 너비를 차지하며, 작업 시간(8시-18시)이 이 1단위 너비에 맞춰 스케일링
# 주말은 1단위 너비를 차지하지만, 작업 시간은 0으로 간주
def map_datetime_to_plot_coord(dt, base_dt, work_start_hour, work_end_hour, daily_work_hours):
    # 기준 날짜를 해당 날의 작업 시작 시간으로 정규화
    base_dt_norm = base_dt.replace(hour=work_start_hour, minute=0, second=0, microsecond=0)
    
    current_dt = base_dt_norm
    plot_coord = 0

    # 기준 날짜부터 현재 날짜까지의 작업 일수를 계산
    while current_dt.date() < dt.date():
        if current_dt.weekday() < 5: # 주중 (월요일=0, 금요일=4)
            plot_coord += daily_work_hours
        current_dt += timedelta(days=1)
        current_dt = current_dt.replace(hour=work_start_hour, minute=0, second=0, microsecond=0)
    
    # 현재 날짜의 작업 시간 추가
    if dt.weekday() < 5: # 주중인 경우에만
        start_of_day_work = dt.replace(hour=work_start_hour, minute=0, second=0, microsecond=0)
        end_of_day_work = dt.replace(hour=work_end_hour, minute=0, second=0, microsecond=0)

        if dt < start_of_day_work:
            # 작업 시작 시간 이전이면 해당 날의 작업 시간은 0
            pass
        elif dt > end_of_day_work:
            # 작업 종료 시간 이후면 해당 날의 전체 작업 시간 추가
            plot_coord += daily_work_hours
        else:
            # 작업 시간 내에 있으면 부분 시간 추가
            plot_coord += (dt - start_of_day_work).total_seconds() / 3600
    
    return plot_coord

def visualize_schedule(data_path):
    # --- 1. 데이터 로딩 및 전처리 ---
    data = pd.read_csv(data_path)
    data["Start_Day"]  = pd.to_datetime(data["Start_Day"])
    data["Finish_Day"] = pd.to_datetime(data["Finish_Day"])
    
    # --- 2. 작업 시간 관련 상수 정의 (simple.py와 동일하게) ---
    WORK_START_HOUR = 8
    WORK_END_HOUR = 16
    DAILY_WORK_HOURS = WORK_END_HOUR - WORK_START_HOUR # 10시간

    # --- 4. 시각화 데이터 준비 ---
    # 전체 스케줄의 가장 빠른 시작 시간을 기준으로 누적 작업 시간 계산
    min_start_day_overall = data['Start_Day'].min().replace(hour=WORK_START_HOUR, minute=0, second=0, microsecond=0)

    # 각 작업의 시작/종료 시간을 누적 작업 시간 좌표로 변환
    data['plot_start'] = data['Start_Day'].apply(lambda x: map_datetime_to_plot_coord(x, min_start_day_overall, WORK_START_HOUR, WORK_END_HOUR, DAILY_WORK_HOURS))
    data['plot_end'] = data['Finish_Day'].apply(lambda x: map_datetime_to_plot_coord(x, min_start_day_overall, WORK_START_HOUR, WORK_END_HOUR, DAILY_WORK_HOURS))
    data['plot_width'] = data['plot_end'] - data['plot_start']

    # 라인 번호 부여
    line_ids = {line: i for i, line in enumerate(sorted(data['Assigned_Line'].unique()))}
    data['line_num'] = data['Assigned_Line'].map(line_ids)

    # --- 5. 시각화 ---
    x_size = (data['Finish_Day'].max() - data['Start_Day'].min()).days
    fig, ax = plt.subplots(figsize=(12, 4 + len(line_ids))) # 그래프 크기 조정

    import itertools
    cmap_list = [ plt.cm.tab20.colors,
                plt.cm.tab20b.colors,
                plt.cm.tab20c.colors,
                plt.cm.Set3.colors,
                plt.cm.Pastel1.colors,
                plt.cm.Accent.colors,
                ]
    # 컬러들을 하나의 리스트로 평탄화
    all_colors = list(itertools.chain(*cmap_list))
    # 고유 스타일에 대해 색 지정
    styles = data["Style"].unique()
    colors = {style: all_colors[i % len(all_colors)] for i, style in enumerate(styles)}

    for idx, row in data.iterrows():
        text_color = 'red' if row['Missed_Delivery'] else 'black' 
        
        rect = patches.Rectangle(
            (row['plot_start'], row['line_num'] - 0.4),  # X 좌표는 누적 작업 시간
            row['plot_width'],                           # 너비는 실제 작업 시간
            0.8,                                         # 높이
            edgecolor='black',
            facecolor= colors[row["Style"]] #color
        )
        ax.add_patch(rect)
        ax.text( row['plot_start'] + row['plot_width'] / 2,
                row['line_num'],
                f"{row['PROD_NO']}",
                va='center', ha='center', fontsize=9, rotation=90, color = text_color)

    # --- 6. X축 설정 (작업 일자 기준으로 커스텀) ---
    max_plot_coord = data['plot_end'].max()

    x_ticks = []
    x_labels = []
    current_date_for_ticks = min_start_day_overall
    current_plot_coord = 0

    # 각 작업 일자의 시작점에 눈금과 레이블 생성
    while current_plot_coord <= max_plot_coord + DAILY_WORK_HOURS:
        if current_date_for_ticks.weekday() < 5: # 주중(월~금)만 눈금 추가
            x_ticks.append(current_plot_coord)
            x_labels.append(current_date_for_ticks.strftime('%Y-%m-%d'))
            current_plot_coord += DAILY_WORK_HOURS
        
        current_date_for_ticks += timedelta(days=1)
        current_date_for_ticks = current_date_for_ticks.replace(hour=WORK_START_HOUR, minute=0, second=0, microsecond=0)

    ax.set_xlim(0, max_plot_coord + DAILY_WORK_HOURS) # X축 범위 설정
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right') # 레이블 회전 및 정렬

    # --- 7. Y축 설정 ---
    ax.set_ylim(-1, len(line_ids))
    ax.set_yticks(list(line_ids.values()))
    ax.set_yticklabels(list(line_ids.keys()))
    ax.set_ylabel("Assigned Line")
    ax.set_title(f"Production Schedule Chart")

    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
