import pandas as pd
from datetime import datetime, timedelta, time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SchedulingEnv(gym.Env):
    def __init__(self, order_path, info_path, start_date_str, daily_work_hours):
        super().__init__()
        self.order_path       = order_path
        self.info_path        = info_path
        self.TODAY            = datetime.strptime(start_date_str, "%Y-%m-%d") #datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        self.WORK_START_HOUR  = 8
        self.WORK_END_HOUR    = self.WORK_START_HOUR + daily_work_hours
        self.DAILY_WORK_HOURS = daily_work_hours

        self._load_data()
        self.info_df = self.original_info_df.copy() # Add this line to initialize info_df
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
        
        # Calculate state size dynamically
        num_lines = len(self.original_info_df['Line No.'].unique())
        # Order info (3) + Line info (4 per line) + Action Mask (num_lines)
        obs_dim = 3 + (num_lines * 4) + num_lines

        # Lower and upper bounds for observation space
        low = np.array([0, -365, 0] + [0, 0, 0, 0] * num_lines + [0] * num_lines, dtype=np.float32)
        high = np.array([100000, 365, 10000] + [365*24*3600, self.DAILY_WORK_HOURS, 10000, 1000000] * num_lines + [1] * num_lines, dtype=np.float32) # 1 for action mask (boolean)
        
        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.get_num_actions())

        self.reset() # Initial reset to set up the environment

    def _load_data(self):
        try:
            self.original_orders_df = pd.read_csv(self.order_path)
            self.original_info_df = pd.read_csv(self.info_path)
            print("데이터 로딩 성공.")
        except FileNotFoundError as e:
            print(f"오류: 파일을 찾을 수 없습니다 - {e.filename}. 파일 경로를 확인해주세요.")
            exit()
        except Exception as e:
            print(f"데이터 로딩 중 오류 발생: {e}")
            exit()

        # 'S/D' 컬럼을 datetime 객체로 변환
        self.original_orders_df['S/D'] = pd.to_datetime(self.original_orders_df['S/D'])

    def _calculate_line_production(self):
        self.line_hourly_production = {}
        for index, row in self.original_info_df.iterrows():
            line_no = row['Line No.']
            daily_prod = row['PRED_SEWING']
            self.line_hourly_production[line_no] = daily_prod / self.DAILY_WORK_HOURS

    def _get_heuristic_best_line(self, order, current_line_availability):
        """
        Replicates the line selection logic from simple.py to find the best line
        for a given order based on the current line availability.
        This method does NOT modify the environment's state.
        """
        prod_no = order['PROD_NO']
        style = order['Style']
        qty = order['Remaining_QTY']
        delivery_date = order['S/D']

        best_line = None
        min_finish_time = datetime.max
        best_line_score = -1

        for line_no, availability in current_line_availability.items():
            line_hourly_prod = self.line_hourly_production[line_no]
            
            temp_start_time = availability['next_available_time']
            # Simulate with a more realistic quantity (e.g., one day's production)
            qty_for_one_day = self.DAILY_WORK_HOURS * line_hourly_prod
            temp_remaining_qty = min(qty, qty_for_one_day)
            temp_current_day_work_hours = availability['current_day_work_hours']
            temp_finish_time = temp_start_time # Initialize to temp_start_time
            
            # Simulate finish time calculation (same as in step method)
            while temp_remaining_qty > 0:
                current_day_date = temp_start_time.date()
                
                if temp_start_time.time() >= time(self.WORK_END_HOUR, 0) or \
                   (temp_start_time.date() > current_day_date and temp_start_time.time() < time(self.WORK_START_HOUR, 0)):
                    temp_start_time = datetime.combine(current_day_date + timedelta(days=1), time(self.WORK_START_HOUR, 0))
                    temp_current_day_work_hours = 0
                    current_day_date = temp_start_time.date()

                while temp_start_time.weekday() >= 5:
                    temp_start_time += timedelta(days=1)
                    temp_start_time = datetime.combine(temp_start_time.date(), time(self.WORK_START_HOUR, 0))
                    temp_current_day_work_hours = 0
                    current_day_date = temp_start_time.date()

                available_hours_today = self.DAILY_WORK_HOURS - temp_current_day_work_hours
                qty_producible_today = int(available_hours_today * line_hourly_prod)

                if temp_remaining_qty <= qty_producible_today:
                    hours_for_this_segment = temp_remaining_qty / line_hourly_prod
                    temp_finish_time = temp_start_time + timedelta(hours=hours_for_this_segment)
                    temp_remaining_qty = 0
                else:
                    temp_finish_time = datetime.combine(current_day_date, time(self.WORK_END_HOUR, 0))
                    temp_remaining_qty -= qty_producible_today
                    temp_start_time = temp_finish_time
                    temp_current_day_work_hours = self.DAILY_WORK_HOURS

            if temp_finish_time.date() >= delivery_date.date():
                continue

            score = 0
            if availability['last_style_processed'] == style:
                score += 100 # High bonus from simple.py

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
        
        return best_line

    def _is_delivery_possible(self, order, start_time):
        """주어진 시작 시간부터 모든 라인을 사용한다고 가정할 때, 주문의 납기 가능 여부를 확인합니다."""
        required_hours = order['Q\'TY'] / sum(self.line_hourly_production.values())
        
        # Calculate available work hours until delivery date
        available_work_hours = 0
        current_day = start_time.date()
        delivery_date = order['S/D'].date()

        while current_day < delivery_date:
            if current_day.weekday() < 5: # Monday to Friday
                available_work_hours += self.DAILY_WORK_HOURS
            current_day += timedelta(days=1)

        return available_work_hours >= required_hours

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Call parent reset for seeding
        # 환경을 초기 상태로 리셋
        self.orders_df = self.original_orders_df.copy()
        self.orders_df['Remaining_QTY'] = self.orders_df['Q\'TY']
        self.info_df = self.original_info_df.copy()

        # 라인 가용성 추적 딕셔너리 초기화
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

        # 주문 정렬: 납기일(S/D) 기준으로 오름차순 정렬 (강화학습에서도 이 순서를 따름)
        self.orders_df = self.orders_df.sort_values(by='S/D').reset_index(drop=True)

        # 납기 가능한 주문과 불가능한 주문을 분리
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

    def _get_state(self):
        # 현재 스케줄링할 주문과 라인들의 상태를 반환
        # Find the first order with Remaining_QTY > 0
        unfinished_orders = self.orders_df[self.orders_df['Remaining_QTY'] > 0]
        if unfinished_orders.empty:
            return None  # 모든 주문이 처리됨

        current_order = unfinished_orders.iloc[0]

        # 상태 벡터 구성 (예시: 실제 구현 시 더 정교하게 구성 필요)
        # [주문_Q'TY, 주문_S/D_남은일수, 라인1_가용시간_초, 라인1_현재일작업시간, 라인1_마지막스타일, ... ]
        state_components = []

        # 1. 현재 주문 정보
        state_components.append(current_order['Remaining_QTY'])
        # 납기일까지 남은 일수 (음수면 납기 지연)
        days_to_delivery = (current_order['S/D'].date() - self.line_availability[list(self.line_availability.keys())[0]]['next_available_time'].date()).days
        state_components.append(days_to_delivery)
        # Style을 숫자로 인코딩 (예: 해시값 또는 One-hot 인코딩)
        # 여기서는 간단히 해시값 사용. 실제로는 고유한 스타일 목록을 만들고 인덱싱하는 것이 좋음.
        state_components.append(hash(current_order['Style']) % 10000) # 간단한 해시값

        # 2. 각 라인의 상태 정보
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

        # Action Mask: 1 if action is allowed, 0 otherwise
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

    def step(self, action):
        # 에이전트의 행동(action)을 받아 환경을 업데이트하고 보상을 계산
        unfinished_orders = self.orders_df[self.orders_df['Remaining_QTY'] > 0]
        if unfinished_orders.empty:
            return self._get_state(), 0, True, False, {}

        current_order_idx = unfinished_orders.index[0]
        current_order = self.orders_df.loc[current_order_idx]
        prod_no = current_order['PROD_NO']
        style = current_order['Style']
        qty = current_order['Remaining_QTY']
        delivery_date = current_order['S/D']
        temp_finish_time = datetime.max # Initialize to a very late date

        line_nos = sorted(list(self.line_availability.keys()))
        assigned_line = line_nos[action] # 에이전트가 선택한 라인

        reward = 0
        done = False
        info = {}

        line_hourly_prod = self.line_hourly_production[assigned_line]

        task_start_time = self.line_availability[assigned_line]['next_available_time']
        remaining_qty = qty
        current_day_work_hours = self.line_availability[assigned_line]['current_day_work_hours']

        final_start_time = task_start_time
        current_segment_start_time = task_start_time
        current_day_date = task_start_time.date() # Initialize current_day_date

        segments = []

        current_day_date = current_segment_start_time.date()

        if current_segment_start_time.time() >= time(self.WORK_END_HOUR, 0) or \
           (current_segment_start_time.date() > current_day_date and current_segment_start_time.time() < time(self.WORK_START_HOUR, 0)):
            current_segment_start_time = datetime.combine(current_day_date + timedelta(days=1), time(self.WORK_START_HOUR, 0))
            current_day_work_hours = 0
            current_day_date = current_segment_start_time.date()

        while current_segment_start_time.weekday() >= 5:
            current_segment_start_time += timedelta(days=1)
            current_segment_start_time = datetime.combine(current_segment_start_time.date(), time(self.WORK_START_HOUR, 0))
            current_day_work_hours = 0
            current_day_date = current_segment_start_time.date()

        available_hours_today = self.DAILY_WORK_HOURS - current_day_work_hours
        qty_producible_today = int(available_hours_today * line_hourly_prod)

        qty_this_segment = 0
        if remaining_qty <= qty_producible_today:
            hours_for_this_segment = remaining_qty / line_hourly_prod
            segment_finish_time = current_segment_start_time + timedelta(hours=hours_for_this_segment)
            current_day_work_hours += hours_for_this_segment
            qty_this_segment = remaining_qty
            remaining_qty = 0
        else:
            segment_finish_time = datetime.combine(current_day_date, time(self.WORK_END_HOUR, 0))
            qty_this_segment = qty_producible_today
            remaining_qty -= qty_producible_today
            current_day_work_hours = self.DAILY_WORK_HOURS
            
        segments.append({
            'start': current_segment_start_time,
            'finish': segment_finish_time,
            'qty_produced': qty_this_segment
        })
        current_segment_start_time = segment_finish_time

        final_finish_time = segments[-1]['finish'] if segments else final_start_time

        # 라인 가용성 업데이트
        self.line_availability[assigned_line]['next_available_time'] = final_finish_time
        self.line_availability[assigned_line]['current_day_work_hours'] = current_day_work_hours if final_finish_time.date() == current_day_date else 0
        self.line_availability[assigned_line]['last_style_processed'] = style
        self.line_availability[assigned_line]['total_assigned_hours'] += (qty / line_hourly_prod) # Update total assigned hours

        # Missed_Delivery 및 Delay_Days 계산 (주문 전체 기준)
        missed_delivery_overall = True if final_finish_time.date() >= delivery_date.date() else False
        delay_days_overall = (final_finish_time.date() - delivery_date.date()).days + 1 if final_finish_time.date() >= delivery_date.date() else 0

        # 스케줄 결과 추가 (각 세그먼트별로)
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

        # Update Remaining_QTY
        self.orders_df.loc[current_order_idx, 'Remaining_QTY'] = remaining_qty

        # --- 보상 계산 ---
        # 납기일 준수 보상
        if remaining_qty == 0: # Only give reward when the order is complete
            if not missed_delivery_overall:
                reward += 10 # 납기일 준수 시 양의 보상
            else:
                reward -= (delay_days_overall * 5) # 납기일 지연 시 큰 음의 보상 (지연 일수에 비례)

        # 스타일 연속성 보상
        if self.line_availability[assigned_line]['last_style_processed'] == style:
            reward += 15 # 스타일 연속성 시 양의 보상 (가중치 상향 조정)

        # 에피소드 종료 여부 확인
        done = self.orders_df['Remaining_QTY'].sum() == 0

        # 최종 보상 (에피소드 종료 시)
        if done:
            # Makespan 계산 (모든 라인의 최종 완료 시간 중 가장 늦은 시간)
            makespan = 0
            for line_no, availability in self.line_availability.items():
                if availability['next_available_time'] > datetime.combine(self.TODAY.date(), time(self.WORK_START_HOUR, 0)): # 작업이 있었다면
                    makespan = max(makespan, (availability['next_available_time'] - datetime.combine(self.TODAY.date(), time(self.WORK_START_HOUR, 0))).total_seconds())

            # Makespan이 짧을수록 높은 보상 (음수 값으로 페널티 부여)
            # Makespan이 길수록 보상이 줄어들도록 설계
            reward -= (makespan / (3600 * 24)) * 0.1 # Makespan 1일당 0.1점 페널티 (조정 필요)

            # 라인 활용도 균등화 보상
            # 모든 라인의 최종 완료 시간의 표준 편차가 작을수록 보상
            finish_times_seconds = []
            for line_no, availability in self.line_availability.items():
                finish_times_seconds.append((availability['next_available_time'] - datetime.combine(self.TODAY.date(), time(self.WORK_START_HOUR, 0))).total_seconds())
            
            if len(finish_times_seconds) > 1:
                std_dev_finish_times = np.std(finish_times_seconds)
                reward -= (std_dev_finish_times / (3600 * 24)) * 1.0 # 표준편차가 클수록 페널티 (가중치 추가 상향 조정)


        next_state = self._get_state()
        return next_state, reward, done, False, info # Add False for truncated

    def get_num_actions(self):
        # 가능한 행동의 개수 (라인의 개수)
        return len(self.info_df['Line No.'].unique())

    def get_scheduled_df(self):
        return pd.DataFrame(self.scheduled_tasks)
