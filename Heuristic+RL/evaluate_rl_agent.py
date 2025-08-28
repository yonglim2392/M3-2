import os
from stable_baselines3 import PPO
from scheduling_env import SchedulingEnv
import numpy as np

def evaluate_agent(model_path, result_path, order_path, info_path, start_date_str, daily_work_hours):
    print("--- 학습된 RL 에이전트 평가 시작 ---")

    if not os.path.exists(model_path):
        print(f"오류: 학습된 모델을 찾을 수 없습니다: {model_path}")
        print("먼저 train_rl_agent.py를 실행하여 모델을 학습시켜야 합니다.")
        return

    # 환경 생성
    env = SchedulingEnv(
        order_path=order_path,
        info_path=info_path,
        start_date_str=start_date_str,
        daily_work_hours=daily_work_hours
    )

    # 학습된 모델 로드
    try:
        model = PPO.load(model_path)
        print(f"모델 로드 성공: {model_path}")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return

    obs, info = env.reset() # reset returns (observation, info)
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True) # deterministic=True for evaluation
        
        # Extract action mask from observation
        num_lines = env.get_num_actions()
        action_mask = obs[-num_lines:] # Action mask is the last part of the observation

        # Check if the predicted action is valid according to the mask
        if action_mask[action] == 0: # If the predicted action is masked (invalid)
            # Find valid actions
            valid_actions = np.where(action_mask == 1)[0]
            if len(valid_actions) > 0:
                # Choose a valid action (e.g., the first valid one, or randomly)
                action = valid_actions[0] # Prioritize the heuristic's best action if available
            else:
                # Fallback if no valid actions are indicated by the mask (should not happen if heuristic is robust)
                action = env.action_space.sample() # Fallback to random action
                print(f"경고: 액션 마스크에 유효한 행동이 없습니다. 무작위 행동으로 대체합니다.")

        # Ensure action is within valid range (redundant if mask is correct, but good safeguard)
        if not (0 <= action < env.action_space.n):
            print(f"경고: 최종 선택된 행동({action})이 유효 범위를 벗어났습니다. 무작위 행동으로 대체합니다.")
            action = env.action_space.sample() # Fallback to random action

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated # Update done status

        total_reward += reward
        obs = next_obs
        step_count += 1

        if obs is None and not done: # 모든 주문이 처리되었으나 done이 False인 경우 (예외 처리)
            print("Warning: Observation is None but done is False. Forcing done=True.")
            done = True

    print("\n--- 평가 에피소드 종료 ---")
    print(f"총 스텝 수: {step_count}")
    print(f"최종 총 보상: {total_reward:.2f}")

    # 스케줄링 결과 확인
    final_schedule_df = env.get_scheduled_df()
    print("\n--- 최종 스케줄링 결과 (일부) ---")
    print(final_schedule_df.head().to_markdown(index=False))
    print(f"총 스케줄된 작업 수: {len(final_schedule_df)}")

    # 결과를 CSV로 저장
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    final_schedule_df.to_csv(result_path, index=False)
    print(f"\n스케줄링 결과가 {result_path} 에 저장되었습니다.")