#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PROGRAM NAME  : scheduling_env.py
PROGRAMMER    : YONG
CREATION DATE : 2025-08-29

[Description]
  - 강화학습(RL) 기반 생산 스케줄링 환경 구현 (Gymnasium 환경)
  - 에이전트가 주문을 어떤 라인에 할당할지를 결정
  - 상태(State): 현재 주문 정보 + 라인별 가용 상태 + 액션 마스크
  - 행동(Action): 각 생산 라인 선택
  - 보상(Reward):
      1. 납기 준수: 주문 완료 시 납기 준수 여부에 따른 보상
      2. 스타일 연속성: 같은 라인에서 같은 스타일 연속 처리 시 보상
      3. 생산 효율성: 라인별 상대 생산량 기반 가산점
      4. Makespan: 전체 생산 완료 시간에 따른 페널티
      5. 라인 활용 균등화: 라인별 완료 시간 표준편차 기반 페널티
  - step() 수행 시 한 주문을 선택된 라인에 할당하고 보상을 계산
  - reset() 수행 시 환경 초기화 및 주문 재정렬
  - 최종 결과는 get_scheduled_df()로 확인 가능
  - 주말 및 근무시간(일별) 고려하여 작업 세그먼트별 시뮬레이션 수행
"""

import pandas as pd
from datetime import datetime, timedelta, time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SchedulingEnv(gym.Env):
    """
    강화학습 스케줄링 환경 (Gymnasium 기반)
    - 에이전트가 주문을 어떤 라인에 할당할지를 선택
    - 보상은 납기 준수, 스타일 연속성, 라인 효율, makespan 균형 등을 고려
    """
    def __init__(self, order_path, info_path, start_date_str, daily_work_hours):
        super().__init__()
        # --- 기본 설정 ---
        self.order_path       = order_path
        self.info_path        = info_path
        self.TODAY            = datetime.strptime(start_date_str, "%Y-%m-%d") # 시작일자
        self.WORK_START_HOUR  = 8                                             # 하루 시작 시간 (08시)
        self.WORK_END_HOUR    = self.WORK_START_HOUR + daily_work_hours       # 하루 종료 시간
        self.DAILY_WORK_HOURS = daily_work_hours                              # 하루 작업 시간

        # --- 데이터 불러오기 및 라인 생산능력 계산 ---
        self._load_data()
        self.info_df = self.original_info_df.copy()  # Info DataFrame 초기화
        self._calculate_line_production()

        # Define observation and action spaces
        # Observation space: [Q'TY, Days_to_Delivery, Style_Hash, Line1_Time_Diff, Line1_Work_Hours, Line1_Style_Hash, ...]
        # Q'TY: 0 to large number (e.g., 100000)
        # Days_to_Delivery: -365 to 365 (approx)
        # Style_Hash: 0 to 10000 (based on modulo)
        # Line_Time_Diff: 0 to very large number (seconds from TODAY 00:00:00, e.g., 365 days * 24 hours * 3600 seconds)
        # Line_Work_Hours: 0 to DAILY_WORK_HOURS
        # Line_Style_Hash: 0 to 10000

        # Estimate observation space size based on _get_state() output
        # Temporarily reset to get a state for space definition
        temp_orders_df = self.original_orders_df.copy()
        temp_orders_df = temp_orders_df.sort_values(by='S/D').reset_index(drop=True)
        
        # --- 상태/행동 공간 정의 ---
        # 상태: 주문(수량, 납기, 스타일) + 라인 상태(가용시간, 작업시간, 마지막 스타일 등) + 액션마스크
        num_lines = len(self.original_info_df['Line No.'].unique())
        obs_dim = 3 + (num_lines * 4) + num_lines # Order info (3) + Line info (4 per line) + Action Mask (num_lines)

        # 관측값 범위 정의 (low/high)
        low = np.array([0, -365, 0] + [0, 0, 0, 0] * num_lines + [0] * num_lines, dtype=np.float32)
        high = np.array([100000, 365, 10000] + [365*24*3600, self.DAILY_WORK_HOURS, 10000, 1000000] * num_lines + [1] * num_lines, dtype=np.float32) # 1 for action mask (boolean)
        
        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.get_num_actions())

        self.reset() # 환경 초기화

    # -------------------------------
    # 데이터 로딩
    # -------------------------------
    def _load_data(self):
        try:
            self.original_orders_df = pd.read_csv(self.order_path) # 주문 데이터
            self.original_info_df = pd.read_csv(self.info_path)    # 라인별 정보
            print("데이터 로딩 성공.")
        except FileNotFoundError as e:
            print(f"오류: 파일을 찾을 수 없습니다 - {e.filename}. 파일 경로를 확인해주세요.")
            exit()
        except Exception as e:
            print(f"데이터 로딩 중 오류 발생: {e}")
            exit()

        # 'S/D'(납기일) 컬럼을 datetime으로 변환
        self.original_orders_df['S/D'] = pd.to_datetime(self.original_orders_df['S/D'])

    # -------------------------------
    # 라인별 시간당 생산량 계산
    # -------------------------------
    def _calculate_line_production(self):
        self.line_hourly_production = {}
        for index, row in self.original_info_df.iterrows():
            line_no = row['Line No.']
            daily_prod = row['PRED_SEWING']
            self.line_hourly_production[line_no] = daily_prod / self.DAILY_WORK_HOURS

    # -------------------------------
    # 휴리스틱 기반 "가장 좋은 라인" 탐색
    # -------------------------------
    def _get_heuristic_best_line(self, order, order_idx, current_line_availability):
        """
        - 납기, 스타일 연속성, 라인 효율 등을 기준으로
        - 현재 주문을 넣기에 가장 적합한 라인을 찾음
        """
        prod_no = order['PROD_NO']
        style = order['Style']
        qty = order['Remaining_QTY']
        delivery_date = pd.to_datetime(order['S/D'])
        order_type = order['TYPE']

        candidate_lines = []
        # --- 라인별 시뮬레이션 ---
        for line_no, availability in current_line_availability.items():
            # 라인이 해당 타입(order_type)을 생산할 수 있는 경우만 고려
            if order_type in self.line_type_production[line_no] and self.line_type_production[line_no][order_type]:
                line_daily_prod = self.line_type_production[line_no][order_type]
                start_time = availability['next_available_time']
                
                sim_current_time = start_time
                sim_remaining_qty = qty
                temp_finish_time = sim_current_time

                # 남은 수량 다 소진될 때까지 시뮬레이션
                while sim_remaining_qty > 0:
                    # 근무시간/주말 체크 후 다음 작업 가능 시간으로 이동
                    if sim_current_time.time() >= time(self.WORK_END_HOUR, 0):
                        sim_current_time = datetime.combine(sim_current_time.date() + timedelta(days=1), time(self.WORK_START_HOUR, 0))
                    elif sim_current_time.time() < time(self.WORK_START_HOUR, 0):
                        sim_current_time = datetime.combine(sim_current_time.date(), time(self.WORK_START_HOUR, 0))

                    sim_current_date = sim_current_time.date()
                    
                    if sim_current_date.weekday() >= 5: # 주말 skip
                        sim_current_time = datetime.combine(sim_current_date + timedelta(days=1), time(self.WORK_START_HOUR, 0))
                        continue

                    # 해당 날짜의 생산능력 가져오기
                    day_index = (sim_current_date - self.TODAY.date()).days

                    if day_index < 0:
                        daily_prod = 0
                    elif day_index >= len(line_daily_prod):
                        daily_prod = line_daily_prod[-1] if line_daily_prod else 0
                    else:
                        daily_prod = line_daily_prod[day_index]

                    if daily_prod <= 0:
                        sim_current_time = datetime.combine(sim_current_date + timedelta(days=1), time(self.WORK_START_HOUR, 0))
                        continue
                    
                    line_hourly_prod = daily_prod / self.DAILY_WORK_HOURS

                    end_of_day = datetime.combine(sim_current_date, time(self.WORK_END_HOUR, 0))
                    available_hours_today = (end_of_day - sim_current_time).total_seconds() / 3600
                    prod_possible_today = available_hours_today * line_hourly_prod

                    # 오늘 하루 안에 끝낼 수 있는 경우
                    if sim_remaining_qty <= prod_possible_today:
                        hours_to_finish = sim_remaining_qty / line_hourly_prod
                        temp_finish_time = sim_current_time + timedelta(hours=hours_to_finish)
                        sim_remaining_qty = 0
                    else:
                        sim_remaining_qty -= prod_possible_today
                        sim_current_time = datetime.combine(sim_current_date + timedelta(days=1), time(self.WORK_START_HOUR, 0))

                # 후보 라인 기록
                candidate_lines.append({
                    'line_no': line_no,
                    'start_time': start_time,
                    'finish_time': temp_finish_time,
                    'daily_prod': line_daily_prod,
                    'last_style': availability['last_style_processed']
                })

        # --- 납기일 안에 끝낼 수 있는 후보만 남김 ---
        valid_candidates = [c for c in candidate_lines if c['finish_time'].date() < delivery_date.date()]
        if not valid_candidates:
            return None

        # --- 각 라인 점수 계산 (납기, 스타일, 효율성 등) ---
        # (여기서 사람이 정한 휴리스틱 규칙이 적용)
        
        for c in valid_candidates:
            day_index = (c['start_time'].date() - self.TODAY.date()).days
            line_daily_prod = c['daily_prod']
            
            prod_on_start_day = 0
            if line_daily_prod:
                if 0 <= day_index < len(line_daily_prod):
                    prod_on_start_day = line_daily_prod[day_index]
                elif day_index >= len(line_daily_prod):
                    prod_on_start_day = line_daily_prod[-1]
            c['prod_on_start_day'] = prod_on_start_day

        # 모든 후보 라인 중 최소/최대 생산량 계산
        prods = [c['prod_on_start_day'] for c in valid_candidates if c['prod_on_start_day'] > 0]
        
        min_prod = min(prods) if prods else 0
        max_prod = max(prods) if prods else 0

        # 스타일 및 효율성 점수 계산
        for c in valid_candidates:
            assigned_lines_for_style = self.style_assignments.get(style, set())
            
            if c['last_style'] == style:
                c['style_score'] = 2  # Best case: 라인의 마지막 작업과 스타일이 완벽히 연속될 때
            elif assigned_lines_for_style and c['line_no'] not in assigned_lines_for_style:
                c['style_score'] = 0  # Worst case: 할당 시 스타일 분산이 발생할 때
            else:
                c['style_score'] = 1  # Neutral: 스타일이 연속되지는 않지만, 새로운 스타일을 할당하여 분산을 일으키지 않을 때
            
            efficiency_score = 0
            if c['prod_on_start_day'] > 0 and prods:
                if max_prod > min_prod:
                    efficiency_score = 1 + ((c['prod_on_start_day'] - min_prod) / (max_prod - min_prod)) * (self.config['reward_weights']['efficiency_max_score'] - 1)
                else:
                    efficiency_score = self.config['reward_weights']['efficiency_max_score']
            c['efficiency_score'] = efficiency_score

        # --- 우선순위에 따라 정렬 ---
        def get_sort_key(c):
            key_tuple = []
            for priority in self.config['heuristic_priority']:
                if priority == 'style':
                    key_tuple.append(c['style_score'])
                elif priority == 'efficiency':
                    key_tuple.append(c['efficiency_score'])
                elif priority == 'finish_time':
                    key_tuple.append(-c['finish_time'].timestamp())
            return tuple(key_tuple)
            
        # 최종 후보 선택
        best_candidate = sorted(valid_candidates, key=get_sort_key, reverse=True)[0]
        return best_candidate['line_no']

    # -------------------------------
    # 납기 가능 여부 체크
    # -------------------------------
    def _is_delivery_possible(self, order, start_time):
        """주어진 주문이 전체 라인 투입 기준으로 납기일 내에 끝낼 수 있는지 확인"""
        required_hours = order['Q\'TY'] / sum(self.line_hourly_production.values())
        
        # Calculate available work hours until delivery date
        available_work_hours = 0
        current_day = start_time.date()
        delivery_date = order['S/D'].date()

        while current_day < delivery_date:
            if current_day.weekday() < 5: # 평일만 계산
                available_work_hours += self.DAILY_WORK_HOURS
            current_day += timedelta(days=1)

        return available_work_hours >= required_hours

    # -------------------------------
    # 환경 리셋
    # -------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Call parent reset for seeding
        # 주문/라인 상태 초기화
        self.orders_df = self.original_orders_df.copy()
        self.orders_df['Remaining_QTY'] = self.orders_df['Q\'TY']
        self.info_df = self.original_info_df.copy()

        # 라인 가용성 초기화
        self.line_availability = {}
        for line_no in self.info_df['Line No.']:
            self.line_availability[line_no] = {
                'next_available_time': datetime.combine(self.TODAY.date(), time(self.WORK_START_HOUR, 0)),
                'current_day_work_hours': 0,
                'last_style_processed': None,
                'total_assigned_hours': 0 # New field for total assigned hours
            }

        self.scheduled_tasks = []
        self.current_order_idx = 0

        # 주문 정렬 (납기일 기준)
        self.orders_df = self.orders_df.sort_values(by='S/D').reset_index(drop=True)

        # 납기 불가능 주문은 뒤로 배치
        possible_orders = []
        impossible_orders = []
        start_time = datetime.combine(self.TODAY.date(), time(self.WORK_START_HOUR, 0))

        for index, order in self.orders_df.iterrows():
            if self._is_delivery_possible(order, start_time):
                possible_orders.append(order)
            else:
                impossible_orders.append(order)

        # 데이터프레임으로 변환 후 정렬
        possible_df = pd.DataFrame(possible_orders)
        impossible_df = pd.DataFrame(impossible_orders)

        if not possible_df.empty:
            possible_df = possible_df.sort_values(by='S/D')
        if not impossible_df.empty:
            impossible_df = impossible_df.sort_values(by='S/D')

        # 두 데이터프레임을 합쳐서 새로운 주문 목록 생성
        self.orders_df = pd.concat([possible_df, impossible_df], ignore_index=True)

        # 초기 상태 반환
        return self._get_state(), {}

    # -------------------------------
    # 상태 벡터 생성
    # -------------------------------
    def _get_state(self):
        """현재 주문 + 라인 상태 + 액션마스크"""
        unfinished_orders = self.orders_df[self.orders_df['Remaining_QTY'] > 0]
        if unfinished_orders.empty:
            return None  # 모든 주문이 처리됨

        current_order = unfinished_orders.iloc[0]
        state_components = []

         # --- 주문 정보 ---
        state_components.append(current_order['Remaining_QTY'])
        # 납기일까지 남은 일수 (음수면 납기 지연)
        days_to_delivery = (current_order['S/D'].date() - self.line_availability[list(self.line_availability.keys())[0]]['next_available_time'].date()).days
        state_components.append(days_to_delivery)
        # Style을 숫자로 인코딩 (예: 해시값 또는 One-hot 인코딩)
        state_components.append(hash(current_order['Style']) % 10000)

         # --- 라인 상태 ---
        for line_no in sorted(self.line_availability.keys()): # 일관된 순서를 위해 정렬
            availability = self.line_availability[line_no]
            # 다음 가용 시간 (초 단위로 변환하여 상대적인 값으로)
            # 기준 시점으로부터의 차이 (예: TODAY 00:00:00 부터의 초)
            time_diff_seconds = (availability['next_available_time'] - datetime.combine(self.TODAY.date(), time(0,0))).total_seconds()
            state_components.append(time_diff_seconds)
            state_components.append(availability['current_day_work_hours'])
            # 마지막 스타일도 숫자로 인코딩
            state_components.append(hash(availability['last_style_processed']) % 10000 if availability['last_style_processed'] else 0)
            state_components.append(availability['total_assigned_hours']) # Add total assigned hours

        # --- 액션 마스크 (휴리스틱 기반) ---
        action_mask = np.zeros(self.get_num_actions(), dtype=np.float32)
        heuristic_best_line = self._get_heuristic_best_line(current_order, self.line_availability)
        
        if heuristic_best_line is not None:
            line_nos = sorted(list(self.line_availability.keys()))
            best_line_idx = line_nos.index(heuristic_best_line)
            action_mask[best_line_idx] = 1 # Only allow the heuristic best line
        else:
            # If heuristic finds no valid line (e.g., due to delivery date), allow all lines
            # or handle as a terminal state/penalty
            action_mask[:] = 1 # Allow all for now, consider more sophisticated handling

        state_components.extend(action_mask)

        return np.array(state_components, dtype=np.float32)

    # -------------------------------
    # step: 한 주문을 라인에 할당
    # -------------------------------
    def step(self, action):
        """
        - action: 에이전트가 선택한 라인 번호
        - 보상: 납기 준수, 스타일 연속, 라인 효율성, makespan, 라인 균등성
        """
        # 1) 아직 처리 안 된 주문(Row)만 필터링
        unfinished_orders = self.orders_df[self.orders_df['Remaining_QTY'] > 0]
        if unfinished_orders.empty:
            # 더 이상 처리할 주문이 없으면: 상태 None(또는 최종 상태), 보상 0, 종료 True
            return self._get_state(), 0, True, False, {}

        # 2) 현재 차례의 주문 한 건 선택 (여기서는 '납기 오름차순'으로 정렬되어 있으므로 맨 앞이 가장 급한 주문)
        current_order_idx = unfinished_orders.index[0]
        current_order = self.orders_df.loc[current_order_idx]
        prod_no = current_order['PROD_NO']                    # 주문 식별자
        style = current_order['Style']                        # 스타일(모델)
        qty = current_order['Remaining_QTY']                  # 아직 생산해야 할 수량
        delivery_date = current_order['S/D']                  # 납기일(datetime)
        temp_finish_time = datetime.max                       # (옵션) 임시 완료시간 초기값(매우 늦은 시간)

        # 3) 액션(정수)을 실제 라인 번호로 매핑
        line_nos = sorted(list(self.line_availability.keys()))
        assigned_line = line_nos[action] # 에이전트가 선택한 라인

         # 4) 보상/종료/부가정보 초기화
        reward = 0
        done = False
        info = {}

        # 5) 선택된 라인의 시간당 생산능력(수량/시간)
        line_hourly_prod = self.line_hourly_production[assigned_line]

        # 6) 해당 라인의 "다음 작업 가능 시작시각"부터 시작
        task_start_time = self.line_availability[assigned_line]['next_available_time']
        remaining_qty = qty                                                                      # 이 주문에서 아직 생산할 남은 수량
        current_day_work_hours = self.line_availability[assigned_line]['current_day_work_hours'] # 오늘 이미 사용한 작업시간(시간 단위)

        # 7) 세그먼트(하루 단위 작업 블록) 시뮬레이션 준비
        final_start_time = task_start_time            # 실제 작업 시작한 시각(첫 세그먼트의 시작)
        current_segment_start_time = task_start_time  # 지금 세그먼트 시작시각(가변)
        current_day_date = task_start_time.date()     # 지금 작업 중인 '날짜'(연-월-일)

        segments = [] # 세그먼트 기록용 리스트(각 세그먼트: 시작, 종료, 그날/그세그먼에서 만든 수량)

        # (안전) 현재 세그먼트의 날짜 캐싱
        current_day_date = current_segment_start_time.date()

        # 8) 근무시간 외에 시작했거나(종료 이후), 날짜가 당일이 아닌 상태면 다음 근무일 08시로 이동
        if current_segment_start_time.time() >= time(self.WORK_END_HOUR, 0) or \
           (current_segment_start_time.date() > current_day_date and current_segment_start_time.time() < time(self.WORK_START_HOUR, 0)):
            current_segment_start_time = datetime.combine(current_day_date + timedelta(days=1), time(self.WORK_START_HOUR, 0))
            current_day_work_hours = 0 # 새로운 근무일이므로 그날 사용시간 리셋
            current_day_date = current_segment_start_time.date()
               
        # 9) 주말(토=5, 일=6)인 경우, 다음 평일 08시로 밀기
        while current_segment_start_time.weekday() >= 5:
            current_segment_start_time += timedelta(days=1)
            current_segment_start_time = datetime.combine(current_segment_start_time.date(), time(self.WORK_START_HOUR, 0))
            current_day_work_hours = 0
            current_day_date = current_segment_start_time.date()

        # 10) 오늘 남은 근무시간과 오늘 생산 가능한 수량 계산
        available_hours_today = self.DAILY_WORK_HOURS - current_day_work_hours # 오늘 남은 시간(시간 단위)
        qty_producible_today = int(available_hours_today * line_hourly_prod)   # 오늘 시간 안에 만들 수 있는 최대 수량(정수화)

        # 11) 오늘(세그먼트)에서 실제로 생산할 수량/종료시각 계산
        qty_this_segment = 0
        if remaining_qty <= qty_producible_today:
            # 남은 수량을 '오늘' 안에 모두 끝낼 수 있음 → 일부 시간만 사용
            hours_for_this_segment = remaining_qty / line_hourly_prod
            segment_finish_time = current_segment_start_time + timedelta(hours=hours_for_this_segment)
            current_day_work_hours += hours_for_this_segment
            qty_this_segment = remaining_qty # 전량 생산
            remaining_qty = 0                # 남은 수량 0
        else:
            # 오늘 가능한 만큼만 만들고, 남은 건 다음 날로 이월
            segment_finish_time = datetime.combine(current_day_date, time(self.WORK_END_HOUR, 0))
            qty_this_segment = qty_producible_today
            remaining_qty -= qty_producible_today
            current_day_work_hours = self.DAILY_WORK_HOURS # 오늘 근무시간 모두 사용

        # 12) 세그먼트 기록(시작/종료/생산수량)
        segments.append({
            'start': current_segment_start_time,
            'finish': segment_finish_time,
            'qty_produced': qty_this_segment
        })

        # 다음 세그먼트는 방금 끝난 시각부터 이어서 시작(보통 다음 날 08시로 정규화하는 로직이 이어짐)
        current_segment_start_time = segment_finish_time

        # 13) 이 주문(혹은 지금까지의 세그먼트)의 최종 완료시각
        final_finish_time = segments[-1]['finish'] if segments else final_start_time

        # ============================
        # (보상) 생산 효율 점수 계산
        #   - 현재 라인의 '시작일' 기준으로 각 라인의 일일 생산량을 비교
        #   - 상대적으로 생산량이 큰 라인을 선택했으면 가산점
        # ============================
        order_type = current_order['TYPE'] # 주문 타입(라인 타입별 생산성 인덱싱용)
        task_start_time = self.line_availability[assigned_line]['next_available_time']
        day_index = (task_start_time.date() - self.TODAY.date()).days # 시작일이 기준일(TODAY)에서 몇 번째 날인지

        prods_for_day = [] # [{'line': 라인번호, 'prod': 그 라인의 해당날 일일예상생산}, ...]
        for line_no in self.line_nos: # ⚠️ self.line_nos는 미리 정의되어 있어야 함
            if order_type in self.line_type_production[line_no]: # 라인이 해당 타입을 생산 가능하면
                line_daily_prod_list = self.line_type_production[line_no][order_type]

                # 해당 시작일(day_index)의 일일 생산량 산출(리스트 범위 밖이면 마지막 값을 사용)
                daily_prod = 0
                if line_daily_prod_list:
                    if 0 <= day_index < len(line_daily_prod_list):
                        daily_prod = line_daily_prod_list[day_index]
                    elif day_index >= len(line_daily_prod_list):
                        daily_prod = line_daily_prod_list[-1] # 범위 밖: 마지막 값으로 대체
                
                if daily_prod > 0:
                    prods_for_day.append({'line': line_no, 'prod': daily_prod})

        # 라인들 간 생산량 상대 비교로 효율 점수 부여
        if len(prods_for_day) > 1:
            min_prod = min(p['prod'] for p in prods_for_day)
            max_prod = max(p['prod'] for p in prods_for_day)

            # 지금 선택한 라인의 해당날 생산량
            assigned_line_prod = next((p['prod'] for p in prods_for_day if p['line'] == assigned_line), 0)

            if assigned_line_prod > 0:
                max_score = 50 # 효율 보상의 최대치
                min_score = 1  # 효율 보상의 최소치

                if max_prod > min_prod:
                    # 선형 스케일링: (선택 라인의 생산량이 최솟값~최댓값 사이 어디쯤인지)에 따라 보상 분배
                    efficiency_score = min_score + ((assigned_line_prod - min_prod) / (max_prod - min_prod)) * (max_score - min_score)
                else:
                    # 모든 라인이 같은 생산량이면 최대 점수
                    efficiency_score = max_score
                
                reward += efficiency_score
        elif len(prods_for_day) == 1 and prods_for_day[0]['line'] == assigned_line:
            # 그 날 해당 타입을 만들 수 있는 라인이 '하나'뿐이고 그 라인을 선택했다면 최고점
            reward += 50
            
        # 14) 선택된 라인의 가용상태 갱신
        self.line_availability[assigned_line]['next_available_time'] = final_finish_time
         # 같은 날 안에서 종료되면, 그 날의 사용 시간 누적 유지 / 날짜가 넘어가면 0으로 리셋
        self.line_availability[assigned_line]['current_day_work_hours'] = (
            current_day_work_hours if final_finish_time.date() == current_day_date else 0
        )
        self.line_availability[assigned_line]['last_style_processed'] = style # 마지막으로 처리한 스타일 갱신
        # 총 할당 시간 누적 (이번 주문 전체 작업시간 = qty / 시간당생산량)
        self.line_availability[assigned_line]['total_assigned_hours'] += (qty / line_hourly_prod) # Update total assigned hours

        # 15) 납기 지연 여부/지연일 계산 (주문 전체 기준)
        #    - 현재 구현은 "완료 날짜가 납기일과 같아도 지연(True)"로 간주(>=)
        missed_delivery_overall = True if final_finish_time.date() >= delivery_date.date() else False
        delay_days_overall = (final_finish_time.date() - delivery_date.date()).days + 1 if final_finish_time.date() >= delivery_date.date() else 0

        # 16) 스케줄링 결과(세그먼트별) 기록: Gantt 나 리포트에 바로 쓰기 좋은 형태 → 추후 Web 표출 고
        for segment in segments:
            self.scheduled_tasks.append({
                'PROD_NO': prod_no,
                'Style': style,
                'Q\'TY': segment['qty_produced'],
                'S/D': delivery_date.strftime('%Y-%m-%d'),
                'Assigned_Line': assigned_line,
                'Start_Day': segment['start'].strftime('%Y-%m-%d %H:%M'),
                'Finish_Day': segment['finish'].strftime('%Y-%m-%d %H:%M'),
                'Missed_Delivery': missed_delivery_overall,
                'Delay_Days': delay_days_overall
            })

        # 17) 주문의 남은 수량 갱신(다 못했으면 다음 step에서 이어서 처리)
        self.orders_df.loc[current_order_idx, 'Remaining_QTY'] = remaining_qty

        # -----------------------------
        # 보상 설계 (Reward Shaping)
        # -----------------------------
        
        # (1) 납기일 준수 보상
        if remaining_qty == 0: # 해당 주문의 생산이 완료된 경우만 보상 평가
            if not missed_delivery_overall:
                reward += 100 # 납기일을 맞췄으면 큰 양의 보상 부여
            else:
                reward -= (delay_days_overall * 100) # 납기일 지연 시 큰 음의 보상 (지연 일수에 비례)

        # (2) 스타일 연속성 보상
        if self.line_availability[assigned_line]['last_style_processed'] == style:
            reward += 50
            # 같은 생산 라인에서 같은 스타일을 연속으로 처리하면
            # 작업 효율성이 증가하므로 추가 보상 부여

        # -----------------------------
        # 에피소드 종료 여부 확인
        # -----------------------------
        done = self.orders_df['Remaining_QTY'].sum() == 0
        # 남은 수량이 모두 0 → 모든 주문 처리가 끝나면 에피소드 종료
        
        # -----------------------------
        # 최종 보상 계산 (에피소드 종료 시)
        # -----------------------------
        if done:
            # (3) Makespan 계산
            # → 전체 생산이 완료되는 데 걸린 총 시간
            makespan = 0
            for line_no, availability in self.line_availability.items():
                if availability['next_available_time'] > datetime.combine(self.TODAY.date(), time(self.WORK_START_HOUR, 0)):
                    # 해당 라인에서 실제 작업이 있었던 경우
                    makespan = max(makespan, (availability['next_available_time'] - datetime.combine(self.TODAY.date(), time(self.WORK_START_HOUR, 0))).total_seconds())
                    
            reward -= (makespan / (3600 * 24)) * 10
            # Makespan이 길수록(생산이 오래 걸릴수록) 페널티 부여
            # 1일당 -10점

            # (4) 라인 활용도 균등화 보상
            # → 모든 라인의 완료 시간이 균등할수록 보상
            finish_times_seconds = []
            for line_no, availability in self.line_availability.items():
                finish_times_seconds.append((availability['next_available_time'] - datetime.combine(self.TODAY.date(), time(self.WORK_START_HOUR, 0))).total_seconds())
            
            if len(finish_times_seconds) > 1:
                std_dev_finish_times = np.std(finish_times_seconds)
                reward -= (std_dev_finish_times / (3600 * 24)) * 1.0
                # 라인별 완료 시간의 표준편차가 클수록 페널티 (작업 분산 불균형을 방지)

        # -----------------------------
        # 다음 상태 반환
        # -----------------------------
        next_state = self._get_state()
        return next_state, reward, done, False, info
        # 반환 값: (상태, 보상, 종료 여부, truncated=False, 추가정보)

     # -------------------------------
    # helper functions
    # -------------------------------
    def get_num_actions(self):
        # 가능한 행동의 개수 (라인의 개수)
        return len(self.info_df['Line No.'].unique())

    def get_scheduled_df(self):
        return pd.DataFrame(self.scheduled_tasks)
