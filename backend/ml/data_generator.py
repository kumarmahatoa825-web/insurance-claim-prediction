import numpy as np
import pandas as pd
import os
from .config import config

def generate_data():
    np.random.seed(config.RANDOM_SEED)
    n = config.NUMBER_OF_ROWS
    
    age = np.clip(np.random.normal(40, 12, n).astype(int), 18, 75)
    gender = np.random.choice(['Male', 'Female', 'Other'], n, p=[0.48, 0.48, 0.04])
    income = np.clip(np.random.lognormal(mean=13, sigma=0.8, size=n).astype(int), 200000, 2000000)
    occupation = np.random.choice(['Salaried', 'Self-Employed', 'Business', 'Retired', 'Student'], n, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    location = np.random.choice(['Urban', 'Suburban', 'Rural'], n, p=[0.5, 0.3, 0.2])
    policy_type = np.random.choice(['Health', 'Vehicle', 'Property', 'Life', 'Travel'], n)
    policy_tenure = np.random.randint(1, 21, n)
    coverage_amount = np.random.randint(100000, 5000001, n)
    
    # Premium derived from coverage and policy_type
    base_premium_rates = {'Health': 0.02, 'Vehicle': 0.05, 'Property': 0.01, 'Life': 0.03, 'Travel': 0.005}
    rates = np.array([base_premium_rates[pt] for pt in policy_type])
    premium = (coverage_amount * rates * np.random.uniform(0.8, 1.2, n)).astype(int)
    
    vehicle_age = np.where(policy_type == 'Vehicle', np.random.randint(0, 21, n), np.random.randint(0, 3, n))
    num_dependents = np.random.randint(0, 6, n)
    
    # previous claims heavily skewed to 0
    previous_claims = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02])
    payment_history = np.clip(np.random.normal(0.8, 0.15, n), 0.0, 1.0)
    renewal_count = np.random.randint(0, 16, n)
    health_score = np.clip(np.random.normal(0.7, 0.2, n), 0.0, 1.0)
    
    # Latent risk process
    # Base factors
    age_factor = (age - 40) / 20.0
    income_factor = -np.log(income) / 15.0
    health_factor = (1.0 - health_score) * 2.0
    history_factor = previous_claims * 0.8
    vehicle_factor = (vehicle_age / 20.0) * (policy_type == 'Vehicle')
    
    # Non-linear combinations
    latent_risk = (
        age_factor * health_factor + 
        history_factor ** 1.2 - 
        payment_history * 1.5 + 
        vehicle_factor * 1.2 +
        (policy_type == 'Health') * health_factor +
        (location == 'Urban') * 0.3
    )
    
    # Add noise
    latent_risk += np.random.normal(0, config.NOISE_LEVEL * 3, n)
    
    # Sigmoid to probability
    prob = 1 / (1 + np.exp(-latent_risk))
    
    # Adjust to target claim rate (approximate shift)
    current_rate = np.mean(prob)
    shift = np.log(config.CLAIM_RATE / (1 - config.CLAIM_RATE)) - np.log(current_rate / (1 - current_rate))
    prob = 1 / (1 + np.exp(-(latent_risk + shift)))
    
    claim_filed = np.random.binomial(1, prob)
    
    # Independent imperfect risk score base
    risk_score_base = np.clip((prob * 100) + np.random.normal(0, 15, n), 0, 100).astype(int)
    
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'income': income,
        'occupation': occupation,
        'location': location,
        'policy_type': policy_type,
        'policy_tenure': policy_tenure,
        'premium': premium,
        'coverage_amount': coverage_amount,
        'vehicle_age': vehicle_age,
        'num_dependents': num_dependents,
        'previous_claims': previous_claims,
        'payment_history': payment_history,
        'renewal_count': renewal_count,
        'health_score': health_score,
        'risk_score_base': risk_score_base,
        'claim_filed': claim_filed
    })
    
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/insurance_claims.csv', index=False)
    print(f"Generated {n} rows. Claim rate: {df['claim_filed'].mean():.4f}")

if __name__ == '__main__':
    generate_data()
