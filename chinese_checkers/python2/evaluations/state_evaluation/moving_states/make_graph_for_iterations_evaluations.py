import pickle
import matplotlib.pyplot as plt
import math

# calculate confidence interval
def calculate_confidence_interval(mean, sample_size, confidence):
    z = 1.96 # z-score for 95% confidence
    std_err = math.sqrt(mean * (1 - mean) / sample_size) # get standard error
    margin_of_error = z * std_err # get margin of error
    confidence_interval = (mean - margin_of_error, mean + margin_of_error) # get confidence interval
    return confidence_interval

if __name__ == "__main__":
    t_res = pickle.load(open('training_evaluation_result.pkl', 'rb'))
    n_res = pickle.load(open('neiboring_evaluation_result.pkl', 'rb'))
    r_res = pickle.load(open('random_evaluation_result.pkl', 'rb'))

    iterations = list(t_res.keys())
    sample_size = 1000

    # seen states
    t_train_true_outcome_accuracy = [x["train_true_accuracy"] for x in t_res.values()]
    t_train_true_outcome_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in t_train_true_outcome_accuracy]

    t_model_train_outcome_accuracy = [x["m_train_accuracy"] for x in t_res.values()]
    t_model_train_outcome_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in t_model_train_outcome_accuracy]

    t_model_true_outcome_accuracy = [x["m_true_accuracy"] for x in t_res.values()]
    t_model_true_outcome_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in t_model_true_outcome_accuracy]

    t_train_action_accuracy = [x["training_policy_accuracy"] for x in t_res.values()]
    t_train_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in t_train_action_accuracy]


    t_search_action_accuracy = [x["search_policy_accuracy"] for x in t_res.values()]
    t_search_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in t_search_action_accuracy]

    t_model_action_accuracy = [x["model_policy_accuracy"] for x in t_res.values()]
    t_model_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in t_model_action_accuracy]

    # neighboring states
    n_search_action_accuracy = [x["search_selection_acc"] for x in n_res.values()]
    n_search_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in n_search_action_accuracy]

    n_model_action_accuracy = [x["model_selection_acc"] for x in n_res.values()]
    n_model_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in n_model_action_accuracy]

    n_model_true_outcome_accuracy = [x["model_vs_true_outcome_accuracy"] for x in n_res.values()]
    n_model_true_outcome_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in n_model_true_outcome_accuracy]

    # random states
    r_search_action_accuracy = [x["search_selection_acc"] for x in r_res.values()]
    r_search_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in r_search_action_accuracy]

    r_model_action_accuracy = [x["model_selection_acc"] for x in r_res.values()]
    r_model_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in r_model_action_accuracy]

    r_model_true_outcome_accuracy = [x["model_vs_true_outcome_accuracy"] for x in r_res.values()]
    r_model_true_outcome_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in r_model_true_outcome_accuracy]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # subplot 1: Training state accuracies
    axs[0, 0].plot(iterations, t_search_action_accuracy, label='Search Accuracy')
    axs[0, 0].fill_between(iterations, [x[0] for x in t_search_action_accuracy_cf], [x[1] for x in t_search_action_accuracy_cf], alpha=0.2)

    axs[0, 0].plot(iterations, t_model_action_accuracy, label='Model Accuracy')
    axs[0, 0].fill_between(iterations, [x[0] for x in t_model_action_accuracy_cf], [x[1] for x in t_model_action_accuracy_cf], alpha=0.2)

    axs[0, 0].plot(iterations, t_train_action_accuracy, label='Train Accuracy')
    axs[0, 0].fill_between(iterations, [x[0] for x in t_train_action_accuracy_cf], [x[1] for x in t_train_action_accuracy_cf], alpha=0.2)
    axs[0, 0].set_xlabel('Iterations')
    axs[0, 0].set_ylabel('Action Accuracy')
    axs[0, 0].set_title('Training State')
    axs[0, 0].legend()

    # subplot 2: Neighboring state accuracies
    axs[0, 1].plot(iterations, n_search_action_accuracy, label='Search Accuracy')
    axs[0, 1].fill_between(iterations, [x[0] for x in n_search_action_accuracy_cf], [x[1] for x in n_search_action_accuracy_cf], alpha=0.2)

    axs[0, 1].plot(iterations, n_model_action_accuracy, label='Model Accuracy')
    axs[0, 1].fill_between(iterations, [x[0] for x in n_model_action_accuracy_cf], [x[1] for x in n_model_action_accuracy_cf], alpha=0.2)
    axs[0, 1].set_xlabel('Iterations')
    axs[0, 1].set_ylabel('Action Accuracy')
    axs[0, 1].set_title('Neighboring State')
    axs[0, 1].legend()

    # subplot 3: Random state accuracies
    axs[1, 0].plot(iterations, r_search_action_accuracy, label='Search Accuracy')
    axs[1, 0].fill_between(iterations, [x[0] for x in r_search_action_accuracy_cf], [x[1] for x in r_search_action_accuracy_cf], alpha=0.2)

    axs[1, 0].plot(iterations, r_model_action_accuracy, label='Model Accuracy')
    axs[1, 0].fill_between(iterations, [x[0] for x in r_model_action_accuracy_cf], [x[1] for x in r_model_action_accuracy_cf], alpha=0.2)
    axs[1, 0].set_xlabel('Iterations')
    axs[1, 0].set_ylabel('Action Accuracy')
    axs[1, 0].set_title('Random State')
    axs[1, 0].legend()

    # subplot 4: All state accuracies
    axs[1, 1].plot(iterations, t_search_action_accuracy, label='Training Search Accuracy')
    axs[1, 1].plot(iterations, n_search_action_accuracy, label='Neighboring Search Accuracy')
    axs[1, 1].plot(iterations, r_search_action_accuracy, label='Random Search Accuracy')
    axs[1, 1].plot(iterations, t_model_action_accuracy, label='Training Model Accuracy')
    axs[1, 1].plot(iterations, n_model_action_accuracy, label='Neighboring Model Accuracy')
    axs[1, 1].plot(iterations, r_model_action_accuracy, label='Random Model Accuracy')
    axs[1, 1].set_xlabel('Iterations')
    axs[1, 1].set_ylabel('Action Accuracy')
    axs[1, 1].set_title('All States')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('../../results/search_accuracy_on_iteration.png')
    plt.clf()

    # plot outcome accuracy
    plt.plot(iterations, t_model_true_outcome_accuracy, label='Training Model Accuracy')
    # plt.fill_between(iterations, [x[0] for x in t_model_true_outcome_accuracy_cf], [x[1] for x in t_model_true_outcome_accuracy_cf], alpha=0.2)

    plt.plot(iterations, t_train_true_outcome_accuracy, label='Training Train Accuracy')
    # plt.fill_between(iterations, [x[0] for x in t_train_true_outcome_accuracy_cf], [x[1] for x in t_train_true_outcome_accuracy_cf], alpha=0.2)

    plt.plot(iterations, n_model_true_outcome_accuracy, label='Neighboring Model Accuracy')
    # plt.fill_between(iterations, [x[0] for x in n_model_true_outcome_accuracy_cf], [x[1] for x in n_model_true_outcome_accuracy_cf], alpha=0.2)

    plt.plot(iterations, r_model_true_outcome_accuracy, label='Random Model Accuracy')
    # plt.fill_between(iterations, [x[0] for x in r_model_true_outcome_accuracy_cf], [x[1] for x in r_model_true_outcome_accuracy_cf], alpha=0.2)


    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Outcome Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../../results/outcome_accuracy_on_iteration.png')
    plt.clf()