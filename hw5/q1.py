import numpy as np
import pandas as pd
import math

data = [2, 1, -1, -2]

# Mixture Coeff, mean, variance
model_1 = [
    (0.5, -1, 1),
    (0.5, 1, 1)
]

model_2 = [
    (0.5, -1.3448246580536993, 0.6914466390907508),
    (0.5, 1.3448246580536996, 0.6914466390907507)
]

def gaussian_prob(x, mean, variance):
    return (math.exp(-((x - mean) ** 2) / (2 * variance)) / math.sqrt(2 * math.pi * variance))

def log_likelihood(data : int, model_vals):
    output = 0
    for mixture, mean, variance in model_vals:
        gaussian_dist = mixture * gaussian_prob(data, mean, variance)
        output += gaussian_dist
    
    return math.log(output)

def q1_1(model_vals):
    total_log = 0
    for value in data:
        log_likelihood_val = log_likelihood(value, model_vals)
        total_log += log_likelihood_val

        print(f"Data point: {value:.4f}, Log-likelihood: {log_likelihood_val:.4f}")
    
    print(f"Total Log-likelihood: {total_log:.4f}")

def q1_2(model_vals, print_flag):
    weight1, mean1, variance1 = model_vals[0]
    weight2, mean2, variance2 = model_vals[1]

    responsibility = []

    for value in data:
        pdf1 = gaussian_prob(value, mean1, variance1)
        pdf2 = gaussian_prob(value, mean2, variance2)

        total_prob = (weight1 * pdf1) + (weight2 * pdf2)
        
        gamma1 = (weight1 * pdf1) / total_prob
        gamma2 = (weight2 * pdf2) / total_prob

        if print_flag: print(f"Data point: {value:.4f}, gamma1 {gamma1:.4f}, gamma2 {gamma2:.4f}")

        responsibility.append([gamma1, gamma2])
    
    return responsibility

def q1_3():
    responsibility = q1_2(model_1, False)
    resp_df = pd.DataFrame(responsibility, columns=["Gamma 0", "Gamma 1"])
    resp_0 = resp_df["Gamma 0"].sum()
    resp_1 = resp_df["Gamma 1"].sum()

    # Weight
    new_weight_0 = resp_0 / len(data)
    new_weight_1 = resp_1 / len(data)

    print(f"New Weight 0: {new_weight_0:.4f}, New Weight 1: {new_weight_1:.4f}")

    # Mean
    mean_0_num = mean_1_num = 0

    for gammas, value in zip(responsibility, data):
        gamma_0, gamma_1 = gammas[0], gammas[1]

        mean_0_num += gamma_0 * value
        mean_1_num += gamma_1 * value

    new_mean_0 = mean_0_num / resp_0
    new_mean_1 = mean_1_num / resp_1

    print(f"New Mean 0: {new_mean_0:.4f}, New Mean 1: {new_mean_1:.4f}")

    # Variance
    var_0_num = var_1_num = 0

    for gammas, value in zip(responsibility, data):
        gamma_0, gamma_1 = gammas[0], gammas[1]

        var_0_num += gamma_0 * (value - new_mean_0) ** 2
        var_1_num += gamma_1 * (value - new_mean_1) ** 2
    
    new_var_0 = var_0_num / resp_0
    new_var_1 = var_1_num / resp_1

    print(f"New Var 0: {new_var_0:.4f}, New Var 1: {new_var_1:.4f}")

if __name__ == "__main__":
    q1_1(model_1)
    print()
    q1_2(model_1, True)
    print()
    q1_3()
    print()
    q1_1(model_2)