import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from scheduling_env import SchedulingEnv

def train_agent(model_path, logs_dir, order_path, info_path, start_date_str, daily_work_hours, train_timesteps):
    
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(f"{model_path}"):
        print("--- 모델이 존재합니다 ---")
    else:
        print("--- 강화학습 에이전트 학습 시작 ---")

        # 환경 생성
        env = SchedulingEnv(order_path=order_path,
                            info_path=info_path,
                            start_date_str=start_date_str,
                            daily_work_hours=daily_work_hours
                            )
        # env = make_vec_env(lambda: SchedulingEnv(order_path=ORDER_PATH, info_path=INFO_PATH), n_envs=1)

        # PPO 모델 정의
        # MlpPolicy는 다층 퍼셉트론(MLP) 신경망을 사용하는 정책입니다.
        # verbose=1은 학습 진행 상황을 출력합니다.
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)

        # 콜백 설정 (학습 중 성능 평가 및 조기 종료)
        # eval_env = SchedulingEnv(order_path=ORDER_PATH, info_path=INFO_PATH)
        # stop_callback = StopTrainingOnNoModelImprovement(max_no_improve_evals=5, min_evals=10, verbose=1)
        # eval_callback = EvalCallback(eval_env, best_model_save_path=model_path,
        #                              log_path=logs_dir, eval_freq=1000,
        #                              deterministic=True, render=False, callback_after_eval=stop_callback)

        # 모델 학습
        # total_timesteps는 에이전트가 환경과 상호작용할 총 스텝 수입니다.
        # 이 값은 문제의 복잡성과 원하는 성능에 따라 조정해야 합니다.
        # 액션 마스크가 상태에 포함되었으므로, 에이전트는 학습을 통해 유효한 행동만 선택하도록 학습해야 합니다.
        print(f"모델 학습 시작: 총 {train_timesteps} timesteps")
        model.learn(total_timesteps=train_timesteps) #, callback=eval_callback)
        print("모델 학습 완료.")

        # 학습된 모델 저장
        model.save(model_path)
        print(f"학습된 모델이 {model_path} 에 저장되었습니다.")
