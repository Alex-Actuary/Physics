import matplotlib.pyplot as plt
import numpy as np


def plot_concentration_snapshot(x, history, times_to_plot=[0, 50, 200, -1]):
  """Affiche quelques profils à des instants précis"""
  plt.figure(figsize=(10, 6))
  for idx in times_to_plot:
    if idx < 0:
      idx = len(history) - 1
    plt.plot(x, history[idx], label=f"Pas de temps {idx}")

  plt.title("Propagation d'un polluant (Convection-Diffusion)", fontsize=14)
  plt.xlabel("Position x", fontsize=12)
  plt.ylabel("Concentration C(x)", fontsize=12)
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.legend()
  plt.show()
