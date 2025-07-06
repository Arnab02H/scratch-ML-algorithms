import matplotlib.pyplot as plt 
### STEP 8: VISUALIZE RESULTS

def plot_results(results):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(results['batch'][2], label='Batch GD', linewidth=2)
    plt.plot(results['sgd'][2], label='Stochastic GD', linewidth=2)
    plt.plot(results['mini_batch'][2], label='Mini-batch GD', linewidth=2)
    plt.xlabel('Iterations/Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves - All Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Zoomed view (last 100 iterations)
    plt.subplot(1, 3, 2)
    plt.plot(results['batch'][2][-100:], label='Batch GD', linewidth=2)
    plt.plot(results['sgd'][2][-100:], label='Stochastic GD', linewidth=2)
    plt.plot(results['mini_batch'][2][-100:], label='Mini-batch GD', linewidth=2)
    plt.xlabel('Last 100 Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Curves - Final Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Final loss comparison
    plt.subplot(1, 3, 3)
    methods = ['Batch GD', 'Stochastic GD', 'Mini-batch GD']
    final_losses = [results['batch'][2][-1], results['sgd'][2][-1], results['mini_batch'][2][-1]]

    plt.bar(methods, final_losses, color=['blue', 'red', 'green'], alpha=0.7)
    plt.xlabel('Method')
    plt.ylabel('Final Loss')
    plt.title('Final Loss Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
