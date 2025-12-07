import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



def scatterplot_inference_vs_accuracy(stats_dfs):
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


def get_optimization_metrics(results_df, tolerance_ms):
    """
    Identifies the minimum latency point and the best point within the tolerance budget.
    """
    # 1. Find the point of minimum REAL latency (Performance)
    min_latency_idx = results_df['avg_latency'].idxmin()
    min_latency_threshold = results_df.loc[min_latency_idx, 'threshold']
    min_latency_val = results_df.loc[min_latency_idx, 'avg_latency']
    min_monetary = results_df.loc[min_latency_idx, 'monetary_cost']

    # 2. Budget Constraint Logic
    target_latency = min_latency_val + (tolerance_ms / 1000.0)

    # Filter: valid latencies, within budget, and at least as much offloading as the fastest point
    budget_candidates = results_df[
        (results_df['avg_latency'] <= target_latency) & 
        (results_df['threshold'] >= min_latency_threshold)
    ]

    if not budget_candidates.empty:
        # Pick the largest threshold (max offloading) that fits the budget
        budget_row = budget_candidates.loc[budget_candidates['threshold'].idxmax()]
        budget_threshold = budget_row['threshold']
        budget_latency = budget_row['avg_latency']
        budget_monetary = budget_row['monetary_cost']
    else:
        budget_threshold = min_latency_threshold
        budget_latency = min_latency_val
        budget_monetary = min_monetary
        
    return {
        'min_threshold': min_latency_threshold,
        'min_latency': min_latency_val,
        'min_cost': min_monetary,
        'budget_threshold': budget_threshold,
        'budget_latency': budget_latency,
        'budget_cost': budget_monetary,
        'target_latency': target_latency
    }

def plot_latency_cost_analysis(results_df, metrics, tolerance_ms, cost_device, cost_cloud):
    """
    Plots the latency and cost curves with annotations for the optimal points.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Unpack metrics for easier reading
    min_thresh = metrics['min_threshold']
    min_lat = metrics['min_latency']
    bud_thresh = metrics['budget_threshold']
    bud_lat = metrics['budget_latency']
    target_lat = metrics['target_latency']

    # 1. Plot the REAL latency curve (Left Axis)
    ax1.plot(results_df['threshold'], results_df['avg_latency'], 
             color='blue', linewidth=2, label='Actual System Latency')
    ax1.set_xlabel('Character Threshold (Jobs <= T go to Device)')
    ax1.set_ylabel('Average System Latency (s)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0.2, max(results_df['avg_latency'].min() * 3.5, 0.6)) 

    # 2. Plot the Cost curve (Right Axis)
    ax2 = ax1.twinx()
    ax2.plot(results_df['threshold'], results_df['monetary_cost'], 
             color='red', linestyle='--', linewidth=2, label='Monetary Cost ($)')
    ax2.set_ylabel('Avg Cost per Request ($)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # 3. Mark the points (on Latency Axis)
    ax1.scatter(min_thresh, min_lat, color='green', s=100, zorder=5, label=f'Fastest (T={min_thresh})')
    ax1.scatter(bud_thresh, bud_lat, color='orange', s=120, marker='P', zorder=6, label=f'Max Budget (T={bud_thresh})')

    # 4. Add reference lines
    ax1.axhline(y=results_df.iloc[0]['avg_latency'], color='gray', linestyle=':', label='All Cloud Baseline')
    ax1.axhline(y=target_lat, color='orange', linestyle='--', alpha=0.3, label=f'Tolerance Limit (+{tolerance_ms}ms)')
    ax1.axvline(x=bud_thresh, color='orange', linestyle='--', alpha=0.5)

    # 5. Annotate
    latency_penalty = bud_lat - min_lat
    ax1.annotate(f"Used Budget:\n+{latency_penalty*1000:.0f}ms / {tolerance_ms}ms",
                 xy=(bud_thresh, bud_lat), 
                 xytext=(bud_thresh - 400, bud_lat),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2", color='orange'))

    # 6. Combine Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    # 7. Add Cost Info Text
    info_text = (f"Cost Settings:\n"
                 f"Device: ${cost_device:.2f}/req\n"
                 f"Cloud:  ${cost_cloud:.2f}/req")
    ax1.text(0.02, 0.95, info_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title(f'Impact of Offloading Threshold on System Latency & Cost\n(Max Tolerance Budget: {tolerance_ms}ms)')
    ax1.grid(True, alpha=0.3)
    plt.show()

def run_full_analysis(df_device, df_cloud, thresholds, tolerance_ms, cost_device, cost_cloud):
    """
    Orchestrates the calculation, metric extraction, and plotting.
    """
    # 1. Calculate Strategy
    results_df = analyze_split_strategy(df_device, df_cloud, thresholds, 
                                        tolerance_ms=tolerance_ms,
                                        cost_device=cost_device,
                                        cost_cloud=cost_cloud)
    
    # 2. Get Metrics
    metrics = get_optimization_metrics(results_df, tolerance_ms)
    
    # 3. Plot
    plot_latency_cost_analysis(results_df, metrics, tolerance_ms, cost_device, cost_cloud)
    
    # 4. Print Analysis
    print(f"--- Performance vs. Budget Analysis ---")
    print(f"1. Fastest Configuration:")
    print(f"   - Threshold: {metrics['min_threshold']} chars")
    print(f"   - Latency:   {metrics['min_latency']:.4f} s")
    print(f"   - Avg Cost:  ${metrics['min_cost']:.4f}")

    print(f"\n2. Max Budget Configuration (Spending up to {tolerance_ms}ms):")
    print(f"   - Threshold: {metrics['budget_threshold']} chars")
    print(f"   - Latency:   {metrics['budget_latency']:.4f} s (+{(metrics['budget_latency']-metrics['min_latency'])*1000:.1f}ms)")
    print(f"   - Avg Cost:  ${metrics['budget_cost']:.4f}")

    savings = metrics['min_cost'] - metrics['budget_cost']
    if savings > 0:
        print(f"\n   -> You save ${savings:.4f} per request by using your latency budget.")
    
    return results_df, metrics



























### functions below here are used for the queueing model and scheduler policy building ###

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import expon
import random






def extract_basic_metrics(df, name="Server"):
    """
    Extracts basic queuing metrics from a raw experiment dataframe.
    Uses columns: 'job_start_ts', 'inference_end_ts', 'inference_time_ms', 'total_latency_ms'
    """
    # Work on a copy to avoid modifying the original dataframe
    df = df.copy()
    df = df.sort_values('job_start_ts')
    
    # 1. Arrival Rate (Lambda)
    # Total time window of the experiment (observed from first arrival to last completion)
    start_time = df['job_start_ts'].min()
    end_time = df['inference_end_ts'].max()
    experiment_duration_sec = (end_time - start_time) / 1000.0
    
    num_requests = len(df)
    arrival_rate = num_requests / experiment_duration_sec if experiment_duration_sec > 0 else 0
    
    # 2. Mean Service Demand (S_bar)
    # inference_time_ms is the pure processing time (service time, no queueing)
    service_times_sec = df['inference_time_ms'] / 1000.0
    mean_service_demand = service_times_sec.mean()
    
    # 3. Empirical Response Time (R)
    # total_latency_ms = inference_end_ts - job_start_ts (includes queueing + service)
    response_times_sec = df['total_latency_ms'] / 1000.0
    
    mean_response_time = response_times_sec.mean()
    p50_response = response_times_sec.median()
    p95_response = response_times_sec.quantile(0.95)
    p99_response = response_times_sec.quantile(0.99)

    # 4. Utilization (rho)
    # Utilization Law: rho = lambda * S
    utilization = arrival_rate * mean_service_demand

    print(f"--- Metrics for {name} ---")
    print(f"  Count:                   {num_requests}")
    print(f"  Duration:                {experiment_duration_sec:.2f} s")
    print(f"  Arrival Rate (λ):        {arrival_rate:.4f} req/s")
    print(f"  Mean Service Demand (S): {mean_service_demand:.4f} s")
    print(f"  Mean Response Time (R):  {mean_response_time:.4f} s")
    print(f"  Response Time P95:       {p95_response:.4f} s")
    print(f"  Utilization (ρ = λ*S):   {utilization:.2%}")
    print("-" * 30)
    
    return {
        'lambda': arrival_rate,
        'mean_service_time': mean_service_demand,
        'mean_response_time': mean_response_time,
        'p95_response_time': p95_response,
        'utilization': utilization
    }





def plot_service_time_distribution(df, title):
    # Extract service times (inference_time_ms) and convert to seconds
    service_times = df['inference_time_ms'] / 1000.0
    mean_service_time = service_times.mean()
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # 1. Plot Histogram & KDE of Empirical Data
    # stat="density" normalizes the histogram so it's comparable to the PDF
    sns.histplot(service_times, bins=200, stat="density", kde=True, 
                 color='skyblue', label='Empirical Data', alpha=0.5, edgecolor='white')
    
    # 2. Plot Theoretical Exponential Distribution
    # The scale parameter for expon is the mean (1/lambda)
    x = np.linspace(0, service_times.max(), 200)
    y = expon.pdf(x, scale=mean_service_time)
    
    plt.plot(x, y, 'r--', lw=3, label=f'Exponential Fit (mean={mean_service_time:.2f}s)')
    
    plt.title(f'Service Time Distribution: {title}')
    plt.xlabel('Service Time (s)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()




def calculate_mg1_response_time(arrival_rate, service_times_sec):
    """
    Calculates Mean Response Time E[R] using the Pollaczek-Khinchine formula.
    E[R] = E[S] + (lambda * E[S^2]) / (2 * (1 - rho))
    """
    if len(service_times_sec) == 0:
        return 0.0, 0.0

    # 1. Estimate Moments from Samples
    E_S = np.mean(service_times_sec)
    E_S2 = np.mean(np.square(service_times_sec)) # Second raw moment E[S^2]
    
    # 2. Utilization
    rho = arrival_rate * E_S
    
    # 3. Stability Check
    if rho >= 1.0:
        return float('inf'), rho
        
    # 4. Pollaczek-Khinchine Formula
    # Mean Waiting Time in Queue
    E_W = (arrival_rate * E_S2) / (2 * (1 - rho))
    
    # Mean Response Time (Service + Queue)
    E_R = E_S + E_W
    
    return E_R, rho

def analyze_routing_mg1(df_device, df_cloud, thresholds):
    """
    Simulates routing based on input character length thresholds using M/G/1 analytical models.
    """
    # 1. Merge Dataframes to align jobs (assuming dataset_item_id matches)
    # We need to know what the service time WOULD be on device vs cloud for every job
    merged_df = pd.merge(
        df_device[['dataset_item_id', 'number_of_characters', 'inference_time_ms']], 
        df_cloud[['dataset_item_id', 'inference_time_ms']], 
        on='dataset_item_id', 
        suffixes=('_dev', '_cloud')
    )
    
    # Convert to seconds
    merged_df['s_dev'] = merged_df['inference_time_ms_dev'] / 1000.0
    merged_df['s_cloud'] = merged_df['inference_time_ms_cloud'] / 1000.0
    
    # 2. Determine Total System Arrival Rate (lambda_total)
    # Estimate from the experiment duration
    start_ts = min(df_device['job_start_ts'].min(), df_cloud['job_start_ts'].min())
    end_ts = max(df_device['inference_end_ts'].max(), df_cloud['inference_end_ts'].max())
    duration_sec = (end_ts - start_ts) / 1000.0
    lambda_total = len(merged_df) / duration_sec
    
    print(f"System Lambda: {lambda_total:.4f} req/s (over {len(merged_df)} jobs)")
    
    results = []
    
    for T in thresholds:
        # --- Traffic Split ---
        # Jobs <= T go to Device, others to Cloud
        mask_device = merged_df['number_of_characters'] <= T
        
        jobs_device = merged_df[mask_device]
        jobs_cloud = merged_df[~mask_device]
        
        # --- Device Queue (M/G/1) ---
        prob_dev = len(jobs_device) / len(merged_df)
        lambda_dev = lambda_total * prob_dev
        
        if not jobs_device.empty:
            r_dev, rho_dev = calculate_mg1_response_time(lambda_dev, jobs_device['s_dev'].values)
        else:
            r_dev, rho_dev = 0.0, 0.0
            
        # --- Cloud Queue (M/G/1) ---
        prob_cloud = len(jobs_cloud) / len(merged_df)
        lambda_cloud = lambda_total * prob_cloud
        
        if not jobs_cloud.empty:
            r_cloud, rho_cloud = calculate_mg1_response_time(lambda_cloud, jobs_cloud['s_cloud'].values)
        else:
            r_cloud, rho_cloud = 0.0, 0.0
            
        # --- System Mean Response Time ---
        # Weighted average of the two queues
        if r_dev == float('inf') or r_cloud == float('inf'):
            system_r = float('inf')
        else:
            system_r = (prob_dev * r_dev) + (prob_cloud * r_cloud)
            
        results.append({
            'threshold': T,
            'avg_latency': system_r,
            'rho_dev': rho_dev,
            'rho_cloud': rho_cloud,
            'prob_offload': prob_cloud # Fraction sent to cloud
        })
        
    return pd.DataFrame(results)




def simulate_routing_validation(df_device, df_cloud, thresholds, num_jobs=10000):
    """
    Discrete-event simulation to validate M/G/1 results.
    Generates Poisson arrivals and samples service times from the empirical data.
    """
    # 1. Merge Dataframes to create a job pool
    # We assume df_device already has the speedup applied to 'inference_time_ms'
    merged_df = pd.merge(
        df_device[['dataset_item_id', 'number_of_characters', 'inference_time_ms', 'job_start_ts', 'inference_end_ts']], 
        df_cloud[['dataset_item_id', 'inference_time_ms', 'job_start_ts', 'inference_end_ts']], 
        on='dataset_item_id', 
        suffixes=('_dev', '_cloud')
    )
    
    # 2. Determine System Arrival Rate (lambda)
    # Use the original timestamps to determine the experiment duration and rate
    start_ts = min(merged_df['job_start_ts_dev'].min(), merged_df['job_start_ts_cloud'].min())
    end_ts = max(merged_df['inference_end_ts_dev'].max(), merged_df['inference_end_ts_cloud'].max())
    duration_sec = (end_ts - start_ts) / 1000.0
    lambda_total = len(merged_df) / duration_sec
    
    print(f"Simulation Lambda: {lambda_total:.4f} req/s | Simulating {num_jobs} jobs per threshold...")

    # Pre-convert pools to seconds for speed
    s_dev_pool = merged_df['inference_time_ms_dev'].values / 1000.0
    s_cloud_pool = merged_df['inference_time_ms_cloud'].values / 1000.0
    char_pool = merged_df['number_of_characters'].values
    pool_size = len(merged_df)

    results = []

    for T in thresholds:
        t_now = 0.0
        server_free_dev = 0.0
        server_free_cloud = 0.0
        total_response_time = 0.0
        
        # Random seed for reproducibility per threshold
        random.seed(42) 

        for _ in range(num_jobs):
            # A. Generate Arrival (Poisson Process)
            inter_arrival = random.expovariate(lambda_total)
            t_now += inter_arrival
            
            # B. Sample a Job (Bootstrap from empirical data)
            idx = random.randint(0, pool_size - 1)
            chars = char_pool[idx]
            s_dev = s_dev_pool[idx]
            s_cloud = s_cloud_pool[idx]
            
            # C. Routing Decision & Queue Simulation
            if chars <= T:
                # Route to Device
                start_service = max(t_now, server_free_dev)
                completion = start_service + s_dev
                server_free_dev = completion
                response = completion - t_now
            else:
                # Route to Cloud
                start_service = max(t_now, server_free_cloud)
                completion = start_service + s_cloud
                server_free_cloud = completion
                response = completion - t_now
            
            total_response_time += response

        avg_latency = total_response_time / num_jobs
        results.append({'threshold': T, 'sim_latency': avg_latency})
        
    return pd.DataFrame(results)