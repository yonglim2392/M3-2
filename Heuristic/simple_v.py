import pandas as pd
from datetime import datetime, timedelta, time

# --- 1. 데이터 로딩 ---
# 실제 파일 경로를 여기에 입력하세요.
# 예: order_path = '/home/yong/AIIS/M3-2/DATA/YONGJIN2_ORDER.csv'
# 예: info_path = '/home/yong/AIIS/M3-2/DATA/YONGJIN2_INFO.csv'
order_path = './DATA/back/YONGJIN2_ORDER.csv'
info_path = './DATA/back/YONGJIN2_INFO.csv'

try:
    orders_df = pd.read_csv(order_path)
    info_df = pd.read_csv(info_path)
    print("데이터 로딩 성공.")
except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다 - {e.filename}. 파일 경로를 확인해주세요.")
    exit()
except Exception as e:
    print(f"데이터 로딩 중 오류 발생: {e}")
    exit()

# --- 3. 스케줄링 초기 설정 ---
# 스케줄 시작일 (오늘 날짜)
# 현재 날짜를 자동으로 가져오려면 datetime.now()를 사용하세요.
# 예시를 위해 고정된 날짜를 사용합니다.
TODAY = datetime(2025, 7, 28)
WORK_START_HOUR = 8
WORK_END_HOUR = 16 # 10시간 작업 (8시 시작 -> 16시 종료)
DAILY_WORK_HOURS = WORK_END_HOUR - WORK_START_HOUR

# --- 2. 데이터 전처리 ---
# 'S/D' 컬럼을 datetime 객체로 변환
orders_df['S/D'] = pd.to_datetime(orders_df['S/D'])

# 라인별 시간당 생산량 계산 (일 생산량 / 8시간)
line_hourly_production = {}
for index, row in info_df.iterrows():
    line_no = row['Line No.']
    daily_prod = row['PRED_SEWING']
    line_hourly_production[line_no] = daily_prod / DAILY_WORK_HOURS



# 라인 가용성 추적 딕셔너리
# 'next_available_time': 라인이 다음 작업을 시작할 수 있는 가장 빠른 시간
# 'current_day_work_hours': 현재 날짜에 해당 라인이 작업한 시간
# 'last_style_processed': 해당 라인이 마지막으로 처리한 Style (작업 연속성 위함)
line_availability = {}
for line_no in info_df['Line No.']:
    line_availability[line_no] = {
        'next_available_time': datetime.combine(TODAY.date(), time(WORK_START_HOUR, 0)),
        'current_day_work_hours': 0,
        'last_style_processed': None
    }

# 스케줄 결과 저장 리스트
scheduled_tasks = []

# --- 4. 스케줄링 로직 ---

# 4.1. 주문 정렬: 납기일(S/D) 기준으로 오름차순 정렬 (최우선 순위)
orders_df = orders_df.sort_values(by='S/D').reset_index(drop=True)

for index, order in orders_df.iterrows():
    prod_no = order['PROD_NO']
    style = order['Style']
    qty = order['Q\'TY']
    delivery_date = order['S/D']

    best_line = None
    min_finish_time = datetime.max # 가장 빠른 완료 시간
    best_line_score = -1 # 라인 선택 점수 (높을수록 좋음)

    # 4.2. 최적 라인 찾기
    # 모든 라인을 순회하며 현재 주문에 가장 적합한 라인 탐색
    for line_no, availability in line_availability.items():
        line_hourly_prod = line_hourly_production[line_no]
        
        # 현재 라인의 가용 시작 시간
        current_line_start_time = availability['next_available_time']
        
        # 이 라인에서 이 주문을 처리하는 데 필요한 총 시간
        total_hours_needed = qty / line_hourly_prod

        temp_start_time = current_line_start_time
        temp_remaining_qty = qty
        temp_current_day_work_hours = availability['current_day_work_hours']
        
        # 예상 완료 시간 계산 (하루 8시간 작업 제한 및 다음날 이어서 작업 고려)
        while temp_remaining_qty > 0:
            current_day_date = temp_start_time.date()
            
            # 만약 현재 작업 시작 시간이 다음 날로 넘어갔다면, 새 날의 시작 시간으로 조정
            if temp_start_time.time() >= time(WORK_END_HOUR, 0) or \
               (temp_start_time.date() > current_day_date and temp_start_time.time() < time(WORK_START_HOUR, 0)):
                temp_start_time = datetime.combine(current_day_date + timedelta(days=1), time(WORK_START_HOUR, 0))
                temp_current_day_work_hours = 0 # 새 날이므로 작업 시간 초기화
                current_day_date = temp_start_time.date() # 날짜 업데이트

            # 주말(토:5, 일:6)이면 다음 월요일로 건너뛰기
            while temp_start_time.weekday() >= 5:
                temp_start_time += timedelta(days=1)
                temp_start_time = datetime.combine(temp_start_time.date(), time(WORK_START_HOUR, 0))
                temp_current_day_work_hours = 0
                current_day_date = temp_start_time.date()

            # 오늘 남은 작업 가능 시간
            available_hours_today = DAILY_WORK_HOURS - temp_current_day_work_hours
            
            # 오늘 생산 가능한 수량
            qty_producible_today = available_hours_today * line_hourly_prod

            if temp_remaining_qty <= qty_producible_today: # 오늘 안에 완료 가능
                hours_for_this_segment = temp_remaining_qty / line_hourly_prod
                temp_finish_time = temp_start_time + timedelta(hours=hours_for_this_segment)
                temp_remaining_qty = 0
            else: # 오늘 안에 완료 불가능, 다음 날로 이월
                temp_finish_time = datetime.combine(current_day_date, time(WORK_END_HOUR, 0))
                temp_remaining_qty -= qty_producible_today
                temp_start_time = temp_finish_time # 다음 작업 시작 시간은 오늘 종료 시간부터
                temp_current_day_work_hours = DAILY_WORK_HOURS # 오늘 작업 시간 모두 소진

        # 예상 완료 시간이 납기일보다 늦으면 이 라인은 부적합 (납기 최우선)
        if temp_finish_time.date() > delivery_date.date():
            continue # 다음 라인으로 넘어감

        # 라인 선택 점수 계산 (높을수록 좋음)
        score = 0
        # 작업 연속성 (Style이 같으면 가산점)
        if availability['last_style_processed'] == style:
            score += 100 # 높은 가산점 부여

        # 라인 사용 균등화 (현재 라인의 다음 가용 시간이 빠를수록 좋음 - 덜 바쁜 라인)
        # 이 부분은 복잡하므로, 일단은 연속성과 납기일 준수에 집중하고,
        # 여러 라인이 납기일 준수 및 연속성 조건을 만족할 경우 가장 빨리 끝나는 라인을 선택하는 방식으로 단순화
        
        # 현재 라인이 더 좋은 후보인지 확인
        # 1. 납기일 준수 (위에서 이미 필터링)
        # 2. 작업 연속성 (가장 높은 점수)
        # 3. 가장 빠른 완료 시간 (동점일 경우)
        
        is_better_candidate = False
        if best_line is None:
            is_better_candidate = True
        elif score > best_line_score:
            is_better_candidate = True
        elif score == best_line_score and temp_finish_time < min_finish_time:
            is_better_candidate = True

        if is_better_candidate:
            best_line = line_no
            min_finish_time = temp_finish_time
            best_line_score = score
            
    # 4.3. 주문 할당 및 라인 가용성 업데이트
    if best_line is None:
        print(f"경고: 주문 {prod_no} (Style: {style}, Q\'TY: {qty})는 납기일 {delivery_date.strftime('%Y-%m-%d')}까지 완료할 수 있는 라인을 찾지 못했습니다.")
        continue # 다음 주문으로 넘어감

    assigned_line = best_line
    line_hourly_prod = line_hourly_production[assigned_line]

    task_start_time = line_availability[assigned_line]['next_available_time']
    remaining_qty = qty
    current_day_work_hours = line_availability[assigned_line]['current_day_work_hours']
    
    final_start_time = task_start_time # 실제 작업 시작 시간 기록
    current_segment_start_time = task_start_time # 현재 세그먼트의 시작 시간

    segments = [] # 여러 날에 걸쳐 작업될 경우 각 세그먼트 저장

    while remaining_qty > 0:
        current_day_date = current_segment_start_time.date()
        
        # 만약 현재 작업 시작 시간이 다음 날로 넘어갔다면, 새 날의 시작 시간으로 조정
        if current_segment_start_time.time() >= time(WORK_END_HOUR, 0) or \
           (current_segment_start_time.date() > current_day_date and current_segment_start_time.time() < time(WORK_START_HOUR, 0)):
            current_segment_start_time = datetime.combine(current_day_date + timedelta(days=1), time(WORK_START_HOUR, 0))
            current_day_work_hours = 0 # 새 날이므로 작업 시간 초기화
            current_day_date = current_segment_start_time.date() # 날짜 업데이트

        # 주말(토:5, 일:6)이면 다음 월요일로 건너뛰기
        while current_segment_start_time.weekday() >= 5:
            current_segment_start_time += timedelta(days=1)
            current_segment_start_time = datetime.combine(current_segment_start_time.date(), time(WORK_START_HOUR, 0))
            current_day_work_hours = 0
            current_day_date = current_segment_start_time.date()

        available_hours_today = DAILY_WORK_HOURS - current_day_work_hours
        qty_producible_today = available_hours_today * line_hourly_prod

        if remaining_qty <= qty_producible_today: # 오늘 안에 완료 가능
            hours_for_this_segment = remaining_qty / line_hourly_prod
            segment_finish_time = current_segment_start_time + timedelta(hours=hours_for_this_segment)
            current_day_work_hours += hours_for_this_segment
            remaining_qty = 0
        else: # 오늘 안에 완료 불가능, 다음 날로 이월
            segment_finish_time = datetime.combine(current_day_date, time(WORK_END_HOUR, 0))
            remaining_qty -= qty_producible_today
            current_day_work_hours = DAILY_WORK_HOURS # 오늘 작업 시간 모두 소진
            
        segments.append({
            'start': current_segment_start_time,
            'finish': segment_finish_time
        })
        current_segment_start_time = segment_finish_time # 다음 세그먼트 시작 시간은 현재 세그먼트 종료 시간부터

    # 최종 완료 시간은 마지막 세그먼트의 종료 시간
    final_finish_time = segments[-1]['finish'] if segments else final_start_time

    # 라인 가용성 업데이트
    line_availability[assigned_line]['next_available_time'] = final_finish_time
    line_availability[assigned_line]['current_day_work_hours'] = current_day_work_hours if final_finish_time.date() == current_day_date else 0 # 다음 날로 넘어가면 0으로 초기화
    line_availability[assigned_line]['last_style_processed'] = style

    # 스케줄 결과 추가
    # Missed_Delivery 및 Delay_Days 계산
    missed_delivery = True if final_finish_time.date() > delivery_date.date() else False
    delay_days = (final_finish_time.date() - delivery_date.date()).days if final_finish_time.date() > delivery_date.date() else 0

    scheduled_tasks.append({
        'PROD_NO': prod_no,
        'Style': style,
        'Q\'TY': qty,
        'S/D': delivery_date.strftime('%Y-%m-%d'),
        'Assigned_Line': assigned_line,
        'Start_Day': final_start_time.strftime('%Y-%m-%d %H:%M'),
        'Finish_Day': final_finish_time.strftime('%Y-%m-%d %H:%M'),
        'Missed_Delivery': missed_delivery,
        'Delay_Days': delay_days
    })

# --- 5. 결과 출력 ---
scheduled_df = pd.DataFrame(scheduled_tasks)
print("\n--- 스케줄링 결과 ---")
print(scheduled_df.to_markdown(index=False))
scheduled_df.to_csv('./RESULT/test.csv', index=False)