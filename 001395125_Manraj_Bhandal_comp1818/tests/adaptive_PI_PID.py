import numpy as np
import math
import pickle
import os


def adaptive_PID_PI(scores, pi_thresholds,pid_thresholds, learning_rate, share, quantile, T_burnin):
    total_scores = len(scores)
    # combines PI and PID thresholds using adaptive weights
    combined_thresholds = np.zeros(total_scores)
    weightPI = 0.5
    weightPID = 0.5

    initial_scores = (scores[0:100])
    absolute_scores = np.abs(initial_scores)
    # Normalisation
    average = np.average(absolute_scores)

    for t in range(T_burnin, total_scores):
        pi_threshold = pi_thresholds[t]
        pid_threshold = pid_thresholds[t]
        pi_weight = weightPI * pi_threshold
        pid_weight = weightPID *   pid_threshold
        combined_thresholds[t] = pi_weight + pid_weight
        # Pinball loss for both controllers
        if scores[t] > pi_threshold :
            pi_loss = quantile * (scores[t] - pi_threshold )

        elif scores[t] <= pi_threshold :
            pi_loss = (1 - quantile) * (pi_threshold  - scores[t])

        if scores[t] >   pid_threshold:
            pid_loss = quantile * (scores[t] -   pid_threshold )

        elif scores[t] <=   pid_threshold:
            pid_loss = (1 - quantile) * (  pid_threshold - scores[t])
        #change controllers weights using normalised losses
        pi_loss_factor = math.exp(-learning_rate * (pi_loss / average))
        pid_loss_factor = math.exp(-learning_rate * (pid_loss / average))
        updated_weightPI = ((weightPI * pi_loss_factor) / (weightPI * pi_loss_factor + weightPID * pid_loss_factor))
        updated_weightPID = ((weightPID * pid_loss_factor) / (weightPI * pi_loss_factor + weightPID * pid_loss_factor))
        weightPI = updated_weightPI
        weightPID = updated_weightPID

        # fixed share
        weightPI = (1 - share) * weightPI + share / 2
        weightPID = (1 - share) * weightPID + share / 2


    return combined_thresholds



def intervals(upper_conformals,lower_conforms,lower_score,upper_score,number_start,number_end):
    # empirical coverage
    adjusted_lower = np.maximum(lower_conforms,-upper_conformals)
    adjusted_higher = np.maximum(upper_conformals, -lower_conforms)

    count_observation = 0
    total_observations =number_end - number_start
    for r in range(number_start,number_end):
        if lower_score[r] <= adjusted_lower[r] and upper_score[r] <= adjusted_higher[r]:
            count_observation += 1

    coverage_value = count_observation / total_observations


    # mean interval width
    interval_width= adjusted_lower + adjusted_higher
    chosen_values = (interval_width[number_start:number_end])
    mean_interval_values=np.average(chosen_values)
    return coverage_value, mean_interval_values


def baselines(baseline,scores,start,end):
 baseline_lower = (np.stack(baseline))[:, 0]
 baseline_upper = (np.stack(baseline))[:, 1]
 lower_score = (scores[1])
 upper_score = (scores[0])


 coverage, width  = intervals(baseline_upper,baseline_lower, lower_score, upper_score, start, end)
 return coverage, width


forecasters = ["ar","transformer","theta","prophet"]
controller_learning_rates = [1,0.1,0.5,0.05,0]
weight_learning_rates=[5,0.5,1,2,10]
share_rates=[0.01,0.05,0.2]
aci_learning_rates = [0.1,0.05,0.01,0.005,0.0001]
adaptive_hyparameter_results = {}
baseline_results ={}
trail_baseline_results = {}
lower_upper_scores = {}
hyperparmeter_results = {}
baseline_results_tuning = {}
fixed_baseline_results = {}
methods = {
     "pi": "Quantile+Integrator (log)",
     "pid": "Quantile+Integrator (log)+Scorecaster",
     "quantile": "Quantile",
     "trail": "Trail",
     "aci": "ACI",
     "aci_clipped": "ACI (clipped)"
}
def get_baseline(dataset,method,controller):
    return dataset[methods[method]][controller]["q"]

with open(os.path.join(os.path.dirname(__file__), "results", 'AMZN.pkl'), 'rb') as o:
    data = pickle.load(o)
for f in forecasters:
    for controller in controller_learning_rates:
        dataset = data[f]
        pi_results = get_baseline(dataset,"pi",controller)
        pid_results= get_baseline(dataset,"pid",controller)
        trail_results = get_baseline(dataset,"trail",None)
        quantile_results = get_baseline (dataset,"quantile",controller)
        aci_results = get_baseline(dataset,"aci",0.1)
        clipped_aci_results = get_baseline(dataset,"aci_clipped",0.1)
        baseline_results[(f,controller)] = (pi_results,pid_results,trail_results,quantile_results,aci_results,clipped_aci_results)



        # upper and lower thresholds for PI/PID
        pi_upper = (np.stack(pi_results))[:, 1]
        pid_upper = (np.stack(pid_results))[:, 1]
        upper_scores = (np.stack(list(dataset["scores"])))[:, 1]
        pi_lower = (np.stack(pi_results))[:, 0]
        pid_lower= (np.stack(pid_results))[:, 0]
        #conformity scores
        lower_scores= (np.stack(list(dataset["scores"])))[:, 0]
        lower_upper_scores[f] = (upper_scores, lower_scores)

        # Testing different weighting learning rates and fixed share values
        for n in weight_learning_rates:
            for a in share_rates:
                results = adaptive_PID_PI(upper_scores, pi_upper, pid_upper, n, a, 0.95, 100)
                results2 = adaptive_PID_PI(lower_scores, pi_lower, pid_lower, n, a, 0.95, 100)
                adaptive_hyparameter_results[(f, controller,n,a)] = (results, results2)




                tuning_coverage, tuning_width = intervals(results, results2, lower_scores, upper_scores, 100, 1100)
                final_coverage2, final_width2 = intervals(results, results2, lower_scores, upper_scores, 1100, 3020)
                hyperparmeter_results[(f,controller, n, a)] = (tuning_coverage, tuning_width, final_coverage2, final_width2)

        run = baseline_results[(f, controller)]
        score_value = lower_upper_scores[f]

        coverage_pi, width_pi = baselines(run[0], score_value, 1100, 3020)
        tunedcoverage_pi, tuned_width_pi = baselines(run[0], score_value, 100, 1100)
        baseline_results_tuning[(f, controller, "pi")] = (tunedcoverage_pi, tuned_width_pi, coverage_pi, width_pi)

        coverage_pid, width_pid = baselines(run[1], score_value, 1100, 3020)
        tuned_coverage_pid, tuned_width_pid = baselines(run[1], score_value, 100, 1100)
        baseline_results_tuning[(f, controller, "pid")] = (tuned_coverage_pid, tuned_width_pid, coverage_pid, width_pid)

        coverage_trail, width_trail = baselines(run[2], score_value, 1100, 3020)
        trail_baseline_results[f] = (coverage_trail, width_trail)

        coverage_quantile, width_quantile = baselines(run[3], score_value, 1100, 3020)
        tuned_coverage_quantile, tuned_width_quantile = baselines(run[3], score_value, 100, 1100)
        baseline_results_tuning[(f, controller, "quantile")] = (tuned_coverage_quantile, tuned_width_quantile,
                                                                coverage_quantile, width_quantile)


    # fixed thresholds baseline
    upper_scores, lower_scores = lower_upper_scores[f]
    scores_counted = len(upper_scores)
    upper_threshold =  np.quantile(upper_scores[100:1100], 0.95)
    lower_threshold =  np.quantile(lower_scores[100:1100], 0.95)
    upper_fixed = [upper_threshold] * scores_counted
    lower_fixed = [lower_threshold] * scores_counted
    upper_fixed = np.array(upper_fixed)
    lower_fixed = np.array(lower_fixed)
    coverage_fixed, width_fixed = intervals(upper_fixed, lower_fixed, lower_scores, upper_scores, 1100, 3020)
    fixed_baseline_results[f] = (coverage_fixed, width_fixed)




    for aci in aci_learning_rates:
        aci_results = get_baseline(dataset, "aci", aci)
        clipped_aci_results = get_baseline(dataset, "aci_clipped", aci)

        aci_coverage, aci_width = baselines(aci_results, lower_upper_scores[f], 1100, 3020)
        tuned_aci_coverage, tuned_aci_width = baselines(aci_results, lower_upper_scores[f], 100, 1100)
        baseline_results_tuning[(f, aci, "aci")] = (tuned_aci_coverage, tuned_aci_width, aci_coverage, aci_width)

        clipped_aci_coverage, clipped_aci_width = baselines(clipped_aci_results, lower_upper_scores[f], 1100, 3020)
        tuned_clipped_aci_coverage, tuned_clipped_aci_width = baselines(clipped_aci_results, lower_upper_scores[f],100, 1100)
        baseline_results_tuning[(f, aci, "aci_clipped")] = (tuned_clipped_aci_coverage, tuned_clipped_aci_width,clipped_aci_coverage, clipped_aci_width)


print("\n Fixed Baseline")
for f in fixed_baseline_results:
    print(f, "Coverage:", fixed_baseline_results[f][0] * 100, "%",
    "Mean Width", fixed_baseline_results[f][1])
chosen_parameters = {}
# Minimum coverage and smallest interval width
print("\n Chosen Parameters and adaptive PI/PID results")
for forecaster in forecasters:
    smallest_tuning_width = None
    final_selected = None
    for key in hyperparmeter_results:
        if key[0] == forecaster:
            if hyperparmeter_results[key][0] >= 0.90:
                     if smallest_tuning_width is None:
                         smallest_tuning_width = hyperparmeter_results[key][1]
                         final_selected = key
                     else:
                         if hyperparmeter_results[key][1] < smallest_tuning_width:
                            smallest_tuning_width= hyperparmeter_results[key][1]
                            final_selected = key


    chosen_parameters[forecaster] = final_selected
    print(forecaster,final_selected,
          "Tuning Coverage:", hyperparmeter_results[final_selected][0] * 100, "%",
          "Tuning Mean Width:", hyperparmeter_results[final_selected][1],
          "Testing Coverage:", hyperparmeter_results[final_selected][2] * 100, "%",
          "Testing Mean Width:", hyperparmeter_results[final_selected][3])


# baseline parameters tuning
final_baselines = {}
print("\n Final Baseline")
for forecaster in forecasters:
    for method in ["pi","pid","quantile","aci","aci_clipped"]:
        smallest_tuning_width = None
        final_selected = None
        for key in baseline_results_tuning:
            if key[0] == forecaster and key[2] == method:
                if baseline_results_tuning[key][0] >= 0.90:
                        if smallest_tuning_width is None:
                           smallest_tuning_width = baseline_results_tuning[key][1]
                           final_selected = key
                        else:
                            if baseline_results_tuning[key][1] < smallest_tuning_width:
                               smallest_tuning_width= baseline_results_tuning[key][1]
                               final_selected = key
        final_baselines[(forecaster, method)] = final_selected
        
        if final_selected is None:
            continue

        print(forecaster, method, final_selected,
        "Coverage:", baseline_results_tuning[final_selected][2] * 100, "%",
        "Mean width:", baseline_results_tuning[final_selected][3])





































































