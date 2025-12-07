
def scatterplot_inference_vs_accuracy(stats_dfs):
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract data for plotting
    inference_times = []
    accuracies = []
    labels = []

    for key, df in stats_dfs.items():
        # Get the "overall" row
        overall_row = df[df['route'] == 'overall']
        
        if not overall_row.empty:
            avg_inference = overall_row['avg_inference_time_ms'].values[0]
            accuracy = overall_row['accuracy_percent'].values[0]
            
            inference_times.append(avg_inference)
            accuracies.append(accuracy)
            
            # Extract model name from filename and subfolder
            subfolder, filename = key.rsplit('/', 1) if '/' in key else ('root', key)
            model_name = filename.replace('stats_experiment_', '').split('_')[0:3]  # adjust based on your naming
            label = f"{subfolder}\n{filename.replace('stats_experiment_', '')[:50]}"  # truncate for readability
            labels.append(label)

    # Create scatter plot
    plt.figure(figsize=(14, 8))
    plt.scatter(inference_times, accuracies, s=100, alpha=0.6, edgecolors='black')

    # Add labels to each point
    for i, label in enumerate(labels):
        plt.annotate(label, (inference_times[i], accuracies[i]), 
                    fontsize=8, ha='right', xytext=(5, 5), 
                    textcoords='offset points', wrap=True)

    plt.xlabel('Avg Inference Time (ms)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Inference Time vs Accuracy Across All Experiments', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_time_measure_distributions(df, model_name):
    import matplotlib.pyplot as plt
    import seaborn as sns

    latency_metrics = {
        'total_latency_ms': [],
        'queueing_time_ms': [],
        'inference_time_ms': []
    }

    # Populate the latency metrics from the DataFrame
    for metric in latency_metrics.keys():
        if metric in df.columns:
            latency_metrics[metric].extend(df[metric].dropna().values)

    # Set the style for seaborn
    sns.set(style="whitegrid")

    # Create subplots for each metric
    plt.figure(figsize=(16, 5))
    
    for i, (metric, data) in enumerate(latency_metrics.items(), start=1):
        plt.subplot(1, len(latency_metrics), i)
        sns.histplot(data, bins=50, kde=True, color=sns.color_palette("husl")[i-1], alpha=0.6, edgecolor='black')
        plt.xlabel(f'{metric} (ms)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Distribution of {metric} for \n {model_name}', fontsize=12, fontweight='bold')
        plt.grid(axis='y', alpha=0.3) 

    plt.tight_layout()
    plt.show()


def plot_time_measures_overlaid(experiment_data, labels):
    import matplotlib.pyplot as plt
    import seaborn as sns
    metrics = ['total_latency_ms', 'queueing_time_ms', 'inference_time_ms']
    
    # Set the style for seaborn
    sns.set(style="whitegrid")

    # Create subplots for each metric
    plt.figure(figsize=(16, 5))
    
    for i, metric in enumerate(metrics, start=1):
        plt.subplot(1, len(metrics), i)  # Create a subplot for each metric
        colors = sns.color_palette("husl", len(experiment_data))  # Use seaborn color palette

        for j, (df, label) in enumerate(zip(experiment_data, labels)):
            if metric in df.columns:
                # Plot histogram with improved aesthetics
                sns.histplot(df[metric].dropna(), bins=50, color=colors[j], label=f'{label}', alpha=0.6, kde=True)

        plt.xlabel(f'{metric} (ms)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title(f'Distribution of {metric} \n Across Experiments', fontsize=12, fontweight='bold')
        plt.legend(title='Experiments', fontsize=12)
        plt.grid(axis='y', alpha=0.5)
        plt.xlim(0,6000)

    plt.tight_layout()
    plt.show()





def plot_characters_vs_inference_time(experiment_data, labels):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Set the style for seaborn
    sns.set(style="whitegrid")

    plt.figure(figsize=(7, 4))

    # Loop through each DataFrame and plot
    for df, label in zip(experiment_data, labels):
        if 'number_of_characters' in df.columns and 'inference_time_ms' in df.columns:
            plt.scatter(df['number_of_characters'], df['inference_time_ms'], alpha=0.5, label=label)

    plt.title('Inference Time vs Number of Input Characters', fontsize=12)
    plt.xlabel('Number of Input Characters', fontsize=12)
    plt.ylabel('Inference Time (ms)', fontsize=12)
    plt.grid()
    plt.legend(title='Experiments', fontsize=10)
    plt.tight_layout()
    plt.show()

    # Calculate and print correlation for each DataFrame
    for df, label in zip(experiment_data, labels):
        correlation = df['number_of_characters'].corr(df['inference_time_ms'])
        print(f'Correlation between number of characters and inference time for {label}: {correlation}')




def fit_mm1_model(df):
    # 1. Calculate Arrival Rate (Lambda)
    # Sort by job start time to ensure correct inter-arrival calculation
    # Changed 'job_start_timestamp' to 'job_start_ts' based on your column output
    df_sorted = df.sort_values('job_start_ts')
    
    # Calculate inter-arrival times in seconds (convert ms to s)
    # Timestamps seem to be in ms based on other columns like 'latency_ms'
    inter_arrival_times = df_sorted['job_start_ts'].diff().dropna() / 1000.0
    mean_inter_arrival = inter_arrival_times.mean()
    lambda_rate = 1.0 / mean_inter_arrival
    
    # 2. Calculate Service Rate (Mu)
    # Service time is inference_time_ms (convert to s)
    service_times = df['inference_time_ms'].dropna() / 1000.0
    mean_service_time = service_times.mean()
    mu_rate = 1.0 / mean_service_time
    
    print(f"--- M/M/1 Fit Results ---")
    print(f"Estimated λ (Arrival Rate): {lambda_rate:.4f} jobs/sec")
    print(f"Estimated μ (Service Rate): {mu_rate:.4f} jobs/sec")
    
    # 3. Check Stability Condition
    rho = lambda_rate / mu_rate
    print(f"Traffic Intensity (ρ): {rho:.4f}")
    
    if rho >= 1:
        print("⚠️ System is unstable (ρ >= 1). The queue will grow infinitely in an M/M/1 model.")
    else:
        # 4. Theoretical M/M/1 Metrics
        theoretical_avg_latency = 1 / (mu_rate - lambda_rate) # W = 1 / (μ - λ)
        theoretical_avg_queue_time = rho / (mu_rate - lambda_rate) # Wq = ρ / (μ - λ)
        
        # 5. Empirical Metrics
        empirical_avg_latency = (df['total_latency_ms'].mean()) / 1000.0
        empirical_avg_queue_time = (df['queueing_time_ms'].mean()) / 1000.0
        
        print(f"\nComparison (Seconds):")
        print(f"Theoretical Avg Total Latency (W): {theoretical_avg_latency:.4f} s")
        print(f"Empirical Avg Total Latency:     {empirical_avg_latency:.4f} s")
        print(f"Theoretical Avg Queue Time (Wq): {theoretical_avg_queue_time:.4f} s")
        print(f"Empirical Avg Queue Time:        {empirical_avg_queue_time:.4f} s")

    return lambda_rate, mu_rate, service_times, inter_arrival_times




import math

def fit_mmc_model(lambda_rate, mu_rates, c=2):
    """
    Fits an M/M/c model for c servers.
    
    Args:
        lambda_rate (float): Total arrival rate (jobs/sec).
        mu_rates (list): List of service rates for each server. 
                         For M/M/c, we usually average them if they differ.
        c (int): Number of servers.
    """
    # Average service rate for the standard M/M/c formula
    avg_mu = sum(mu_rates) / len(mu_rates)
    
    rho = lambda_rate / (c * avg_mu)
    
    print(f"--- M/M/{c} Model Results ---")
    print(f"Combined Arrival Rate (λ): {lambda_rate:.4f}")
    print(f"Avg Service Rate (μ): {avg_mu:.4f}")
    print(f"Traffic Intensity (ρ): {rho:.4f}")

    if rho >= 1:
        print("⚠️ System is unstable (ρ >= 1).")
        return None

    # Calculate Probability of 0 jobs in system (P0)
    sum_part = sum([(c * rho)**n / math.factorial(n) for n in range(c)])
    last_term = ((c * rho)**c / (math.factorial(c) * (1 - rho)))
    p0 = 1 / (sum_part + last_term)

    # Calculate Avg Queue Length (Lq)
    lq = (p0 * (lambda_rate / avg_mu)**c * rho) / (math.factorial(c) * (1 - rho)**2)
    
    # Calculate Avg Waiting Time in Queue (Wq)
    wq = lq / lambda_rate
    
    # Calculate Avg Total Latency (W) = Wq + (1/μ)
    w = wq + (1 / avg_mu)

    print(f"Theoretical Avg Queue Time (Wq): {wq:.4f} s")
    print(f"Theoretical Avg Total Latency (W): {w:.4f} s")
    
    return w, wq



def analyze_split_strategy(device_df, cloud_df, thresholds, tolerance_ms=0, cost_device=0.0, cost_cloud=0.0):
    """
    Analyzes the performance of splitting traffic based on input character length.
    
    Args:
        tolerance_ms: How many milliseconds of extra latency we are willing to tolerate 
                      to keep a job on-device. (Subtracts this from device latency cost).
        cost_device: Monetary/Resource cost per request for on-device.
        cost_cloud: Monetary/Resource cost per request for cloud.
    """
    results = []

    # 1. Combine data to simulate the full incoming stream
    workload_df = device_df.copy()
    
    # Calculate total arrival rate of the system
    workload_df = workload_df.sort_values('job_start_ts')
    inter_arrival_times = workload_df['job_start_ts'].diff().dropna() / 1000.0
    lambda_total = 1.0 / inter_arrival_times.mean()

    for T in thresholds:
        # --- Split the Workload ---
        short_jobs = workload_df[workload_df['number_of_characters'] <= T]
        long_jobs = workload_df[workload_df['number_of_characters'] > T]
        
        # Probability of being short/long
        p_short = len(short_jobs) / len(workload_df)
        p_long = 1.0 - p_short
        
        # Arrival rates for each subsystem
        lambda_device = lambda_total * p_short
        lambda_cloud = lambda_total * p_long
        
        # --- Estimate Service Rates (Mu) ---
        if len(short_jobs) > 0:
            mu_device = 1.0 / (short_jobs['inference_time_ms'].mean() / 1000.0)
        else:
            mu_device = 0 
            
        if len(long_jobs) > 0:
            mu_cloud = 1.0 / (cloud_df['inference_time_ms'].mean() / 1000.0)
        else:
            mu_cloud = 0

        # --- Calculate M/M/1 Metrics ---
        # Device Queue
        if lambda_device > 0 and mu_device > lambda_device:
            W_device = 1.0 / (mu_device - lambda_device)
        elif lambda_device == 0:
            W_device = 0
        else:
            W_device = float('inf') # Unstable
            
        # Cloud Queue
        if lambda_cloud > 0 and mu_cloud > lambda_cloud:
            W_cloud = 1.0 / (mu_cloud - lambda_cloud)
        elif lambda_cloud == 0:
            W_cloud = 0
        else:
            W_cloud = float('inf') # Unstable
            
        # --- Weighted Average System Latency & Costs ---
        if W_device == float('inf') or W_cloud == float('inf'):
            avg_system_latency = float('inf')
            adjusted_cost = float('inf')
            monetary_cost = float('inf')
        else:
            avg_system_latency = p_short * W_device + p_long * W_cloud
            
            # Monetary Cost Calculation
            monetary_cost = p_short * cost_device + p_long * cost_cloud

            # Apply tolerance: We perceive on-device latency as "cheaper" by tolerance_ms
            # Effective Cost = p_short * (W_device - tolerance) + p_long * W_cloud
            perceived_device_latency = max(0, W_device - (tolerance_ms / 1000.0))
            adjusted_cost = p_short * perceived_device_latency + p_long * W_cloud
            
        results.append({
            'threshold': T,
            'avg_latency': avg_system_latency,
            'adjusted_cost': adjusted_cost,
            'monetary_cost': monetary_cost,
            'lambda_device': lambda_device,
            'lambda_cloud': lambda_cloud,
            'p_short': p_short
        })

    return pd.DataFrame(results)