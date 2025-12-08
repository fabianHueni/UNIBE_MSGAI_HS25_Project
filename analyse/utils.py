
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

    plt.tight_layout()
    plt.show()


def plot_inference_time_distribution(experiment_data, labels, model_names):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set the style for seaborn
    sns.set(style="whitegrid")

    fig, axes = plt.subplots(nrows=len(experiment_data), ncols=len(experiment_data[0]), figsize=(14, 10))
    for j, (model_experiments, model_labels, model_name) in enumerate(zip(experiment_data, labels, model_names)):
        colors = sns.color_palette("husl", len(experiment_data))  # Use seaborn color palette

        # take the maximum of all experiments for consistent x limit
        max_x_limit = max([df['inference_time_ms'].quantile(0.98) for df in model_experiments])


        # Loop through each DataFrame and plot
        for i, (df, label) in enumerate(zip(model_experiments, model_labels)):
            sns.histplot(df['inference_time_ms'].dropna(), bins=50, color=colors[j], label=f'{label}', alpha=0.6, kde=True, ax=axes[i][j])

            if i == 0:
                axes[i][j].set_title(model_name, fontsize=12)
                axes[i][j].set_xlabel('', fontsize=12)
            elif i == len(model_experiments) - 1:
                axes[i][j].set_xlabel('Inference Time (ms)', fontsize=12)
            else:
                axes[i][j].set_title('', fontsize=12)
                axes[i][j].set_xlabel('', fontsize=12)
            if j == 0:
                axes[i][j].set_ylabel('Frequency', fontsize=12)
            else:
                axes[i][j].set_ylabel('', fontsize=12)

            axes[i][j].grid()
            axes[i][j].legend(fontsize=10)

            # Set consistent x and y limits for better comparison
            axes[i][j].set_xlim(0, max_x_limit)
            axes[i][j].set_ylim(0, None)

    # remove not used subplots
    fig.delaxes(axes[3][2])
    fig.delaxes(axes[3][3])


    fig.suptitle('Distribution of Inference Time over Model and Devices', fontsize=20)
    plt.tight_layout()

    # save plot as png
    plt.savefig('./plots/inference_time_distribution.png', dpi=300)

    plt.show()




def plot_warm_up_phase(experiment_data, labels, model_names):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set the style for seaborn
    sns.set(style="whitegrid")

    fig, axes = plt.subplots(nrows=len(experiment_data), ncols=len(experiment_data[0]), figsize=(14, 10))

    for j, (model_experiments, model_labels, model_name) in enumerate(zip(experiment_data, labels, model_names)):
        colors = sns.color_palette("husl", len(experiment_data))  # Use seaborn color palette


        # Loop through each DataFrame and plot
        for i, (df, label) in enumerate(zip(model_experiments, model_labels)):
            df_sorted = df.sort_values('job_start_ts').reset_index(drop=True)
            # calculate moving average over 3 samples
            df_sorted['inference_time_ms'] = df_sorted['inference_time_ms'].rolling(window=5).mean()

            first_20 = df_sorted.head(20)
            axes[i][j].plot(first_20.index, first_20['inference_time_ms'], marker='o', linestyle='-', color=colors[i], label=f'{label}')
            axes[i][j].set_title(f'Warmup Effect - {label}')
            axes[i][j].set_xlabel('Sample Index')
            axes[i][j].set_ylabel('Inference Time (ms)')
            axes[i][j].grid(True, alpha=0.3)


            if i == 0:
                axes[i][j].set_title(model_name, fontsize=12)
                axes[i][j].set_xlabel('', fontsize=12)
            elif i == len(model_experiments) - 1:
                axes[i][j].set_xlabel('Inference Time (ms)', fontsize=12)
            else:
                axes[i][j].set_title('', fontsize=12)
                axes[i][j].set_xlabel('', fontsize=12)
            if j == 0:
                axes[i][j].set_ylabel('Frequency', fontsize=12)
            else:
                axes[i][j].set_ylabel('', fontsize=12)

            axes[i][j].grid()
            axes[i][j].legend(fontsize=10)


    # remove not used subplots
    fig.delaxes(axes[3][2])
    fig.delaxes(axes[3][3])


    fig.suptitle('Distribution of Inference Time over Model and Devices', fontsize=20)
    plt.tight_layout()

    # save plot as png
    plt.savefig('./plots/inference_time_distribution.png', dpi=300)

    plt.show()




def plot_characters_vs_inference_time(experiment_data, labels, model_names):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    # Set the style for seaborn
    sns.set(style="whitegrid")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    axes_flat = axes.flatten()
    for ax, model_experiments, model_labels, model_name in zip(axes_flat, experiment_data, labels, model_names):

        # Loop through each DataFrame and plot
        for df, label in zip(model_experiments, model_labels):
            if 'number_of_characters' in df.columns and 'inference_time_ms' in df.columns:
                # generate a size vector to set all sizes to 1 for better visibility
                size = np.ones(len(df)) * 5
                ax.scatter(df['number_of_characters'], df['inference_time_ms'], alpha=0.6, label=label, sizes=size)

        ax.set_title(model_name, fontsize=12)
        ax.set_xlabel('Number of Input Characters', fontsize=12)
        ax.set_ylabel('Inference Time (ms)', fontsize=12)
        ax.grid()
        ax.legend(title='Experiments', fontsize=10)

    fig.suptitle('Inference Time vs. Input Characters', fontsize=20)
    plt.tight_layout()

    # save plot as png
    plt.savefig('./plots/characters_vs_inference_time.png', dpi=300)

    plt.show()

def calc_correlation_characters_inference(experiment_data, labels):
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