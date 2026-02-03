import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# Do wizualizacji
import matplotlib.pyplot as plt

class ICUEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, data=None, max_steps=20, use_real_data=True):
        super(ICUEnv, self).__init__()
        # Stany: tętno, ciśnienie, SOFA, wiek
        self.observation_space = spaces.Box(low=np.array([30, 50, 0, 18]), 
                                            high=np.array([180, 200, 24, 100]), 
                                            dtype=np.float32)
        # Akcje: 0 - brak, 1 - płyny, 2 - wazopresory, 3 - wentylacja
        self.action_space = spaces.Discrete(4)
        self.max_steps = max_steps

        if data is not None:
            self.data = data
        elif use_real_data:
            self.data = self._load_mimic_data()

    def _load_mimic_data(self):
        base = "mimic-iii-clinical-database-demo-1.4/"
        patients = pd.read_csv(base + "PATIENTS.csv")
        icustays = pd.read_csv(base + "ICUSTAYS.csv")
        admissions = pd.read_csv(base + "ADMISSIONS.csv")

        merged = icustays.merge(admissions, on=["subject_id", "hadm_id"], how="left")
        merged = merged.merge(patients, on=["subject_id"], how="left")

        # 1 epizod = 1 pobyt na OIOM
        episodes = []
        for _, row in merged.iterrows():
            # Uproszczone cechy: wiek, los, hospital_expire_flag, icu_los
            age = 2026 - int(str(row["dob"])[:4]) if "dob" in row and pd.notnull(row["dob"]) else 60
            heart_rate = np.random.uniform(60, 120, self.max_steps)
            bp = np.random.uniform(80, 160, self.max_steps)
            sofa = np.random.randint(2, 15, self.max_steps)
            actions = np.random.randint(0, 4, self.max_steps)
            done = [False]*(self.max_steps-1) + [True]
            outcome = [0]*(self.max_steps-1) + [1 if row.get("hospital_expire_flag", 0)==0 else 0]
            episode = pd.DataFrame({
                "heart_rate": heart_rate,
                "bp": bp,
                "sofa": sofa,
                "age": [age]*self.max_steps,
                "action": actions,
                "done": done,
                "outcome": outcome
            })
            episodes.append(episode)
        return episodes

    def _generate_dummy_data(self, num_patients=100):
        episodes = []
        for _ in range(num_patients):
            length = np.random.randint(5, self.max_steps)
            episode = pd.DataFrame({
                "heart_rate": np.random.uniform(60, 120, length),
                "bp": np.random.uniform(80, 160, length),
                "sofa": np.random.randint(2, 15, length),
                "age": np.random.randint(30, 90, length),
                "action": np.random.randint(0, 4, length),
                "done": [False]*(length-1) + [True],
                "outcome": [0]*(length-1) + [np.random.choice([0, 1])]  # 1=survived, 0=death
            })
            episodes.append(episode)
        return episodes

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = np.random.randint(0, len(self.data))
        self.current_patient = self.data[idx]
        self.current_step = 0
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        row = self.current_patient.iloc[self.current_step]
        return np.array([row["heart_rate"], row["bp"], row["sofa"], row["age"]], dtype=np.float32)

    def step(self, action):
        row = self.current_patient.iloc[self.current_step]
        done = bool(row["done"])
        reward = 0.0

        # Nagroda tylko na końcu epizodu
        if done:
            reward = 1.0 if row["outcome"] == 1 else -1.0

        self.current_step += 1
        if self.current_step >= len(self.current_patient):
            done = True

        obs = self._get_obs() if not done else np.zeros(4, dtype=np.float32)
        info = {"real_action": row["action"], "outcome": row["outcome"] if done else None}
        return obs, reward, done, False, info # truncated is always False

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, State: {self._get_obs()}")

    def close(self):
        pass

if __name__ == "__main__":
    print("Testowanie środowiska ICUEnv na danych MIMIC-III (demo)...")
    env = ICUEnv(use_real_data=True)
    num_episodes = 1000
    episodic_rewards = []
    all_obs_hist = []
    all_action_hist = []
    all_reward_hist = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        obs_hist = []
        action_hist = []
        reward_hist = []
        while not done and step < env.max_steps:
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            obs_hist.append(obs)
            action_hist.append(action)
            reward_hist.append(reward)
            obs = next_obs
            total_reward += reward
            step += 1
        episodic_rewards.append(total_reward)
        if ep == 0:
            all_obs_hist = np.array(obs_hist)
            all_action_hist = np.array(action_hist)
            all_reward_hist = np.array(reward_hist)
        print(f"Episod {ep+1}: reward={total_reward}")

    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    axs[0].plot(all_obs_hist[:, 0], label="Tętno")
    axs[0].set_ylabel("Tętno")
    axs[0].legend()
    axs[1].plot(all_obs_hist[:, 1], label="Ciśnienie")
    axs[1].set_ylabel("Ciśnienie")
    axs[1].legend()
    axs[2].plot(all_obs_hist[:, 2], label="SOFA")
    axs[2].set_ylabel("SOFA")
    axs[2].legend()
    axs[3].plot(all_action_hist, label="Akcja")
    axs[3].set_ylabel("Akcja")
    axs[3].set_yticks([0, 1, 2, 3])
    axs[3].set_yticklabels(["brak", "płyny", "wazopresory", "wentylacja"])
    axs[3].legend()
    axs[4].plot(all_reward_hist, label="Nagroda")
    axs[4].set_ylabel("Nagroda")
    axs[4].set_xlabel("Krok")
    axs[4].legend()
    plt.suptitle("Przebieg epizodu ICUEnv (agent losowy) - pierwszy epizod")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(episodic_rewards, bins=np.arange(-1.5, 2, 1), rwidth=0.7)
    plt.xlabel("Nagroda końcowa (przeżycie=1, zgon=-1)")
    plt.ylabel("Liczba epizodów")
    plt.title(f"Podsumowanie wyników {num_episodes} epizodów (agent losowy)")
    plt.xticks([-1, 1], ["zgon", "przeżycie"])
    plt.show()