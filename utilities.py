from signal import siginterrupt
import numpy as np

def generate_revision_times(N, R, T, dt):
    # N: number of agents
    # R: exponential distribution rate
    # T: simulation time
    # dt: step size to discretize the revision times
    num_samples = int(R*T/dt)    # Number of samples to generate per agent
    cond = False                 # Condition to check if the minimum revision time is after the simulation time
    while(not cond):
        rev_times = np.random.exponential(1/R, (N, num_samples))            # Generate revision times
        cumsum_rev_times = (np.cumsum(rev_times, 1)/dt).astype(int)         # Cumulative revision times (in steps)
        min_last_rev_time = np.min(cumsum_rev_times[:, -1]).astype(int)     # Minimum last revision time
        if((min_last_rev_time*dt) >= T):                                    # Checking condition
            cond = True
            _, sufficient_samples = np.where(cumsum_rev_times < int(T/dt))
            min_samples = np.max(sufficient_samples).astype(int)
            cumsum_rev_times = cumsum_rev_times[:, :min_samples+2]
        else:
            print('Regenerating revision times...')
            num_samples = int(2*num_samples)
    return cumsum_rev_times

def get_strategic_distribution(N, n, selected_strategies):
    X = np.zeros((n, 1))
    for i in range(n):
        Xi = np.where(selected_strategies == i)
        X[i, 0] = len(Xi[0])
    return X/N