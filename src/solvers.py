import numpy as np


class ConvectionDiffusionSolver1D:

  def __init__(
      self, L=1.0, T=0.5, Nx=100, Nt=1000, v=1.0, D=0.01, initial_profile="gaussian"
  ):
    self.L = L  # Longueur du domaine
    self.T = T  # Temps final
    self.Nx = Nx  # Nombre de points d'espace
    self.Nt = Nt  # Nombre de pas de temps
    self.v = v  # Vitesse de convection
    self.D = D  # Coefficient de diffusion

    self.dx = L / (Nx - 1)
    self.dt = T / Nt
    self.x = np.linspace(0, L, Nx)

    # Vérification de la condition CFL (stabilité numérique)
    self.cfl_conv = v * self.dt / self.dx
    self.cfl_diff = D * self.dt / (self.dx**2)
    print(f"CFL Convection: {self.cfl_conv:.3f} (doit être < 1)")
    print(f"CFL Diffusion: {self.cfl_diff:.3f} (doit être < 0.5)")

    # Initialisation de la concentration C
    self.C = self._initialize_profile(initial_profile)
    self.history = [self.C.copy()]

  def _initialize_profile(self, profile_type):
    if profile_type == "gaussian":
      # Une cloche gaussienne centrée au début
      return np.exp(-100 * (self.x - 0.3) ** 2)
    elif profile_type == "step":
      # Un créneau (crée des oscillations si le schéma est mal choisi)
      return np.where((self.x >= 0.2) & (self.x <= 0.4), 1.0, 0.0)
    else:
      raise ValueError("Profil inconnu ('gaussian' ou 'step')")

  def solve(self):
    """Résolution par différences finies explicites"""
    for n in range(self.Nt):
      C_new = self.C.copy()

      # Schéma aux différences finies (Amont pour l'espace, Euler explicite en temps)
      for i in range(1, self.Nx - 1):
        # Dérivée convective (décentrée amont si v > 0)
        dC_dx = (self.C[i] - self.C[i - 1]) / self.dx
        # Dérivée diffusive (centrée)
        d2C_dx2 = (self.C[i + 1] - 2 * self.C[i] + self.C[i - 1]) / (
            self.dx**2
        )

        C_new[i] = self.C[i] - self.dt * (self.v * dC_dx - self.D * d2C_dx2)

      self.C = C_new
      self.history.append(self.C.copy())

    return self.x, np.array(self.history)
