"""
SOLUCIÓN DE ECUACIONES DIFERENCIALES CON VALOR EN LA FRONTERA
Requisitos:
    pip install numpy scipy matplotlib

Ejecuta:
    python main.py

Notas:
- En el problema 1 se usa la ecuación físicamente consistente y habitual
  para un cable: y''(x) = -w(x)/T. El enunciado parece mostrar -w(x)T,
  pero multiplicar por T no es consistente dimensionalmente.
- En el problema 2 se interpreta P(x) = A(x)/0.005 + 0.01 tal como aparece
  en el enunciado.
- En el problema 5 puedes modificar los parámetros que solicita el programa.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.integrate import solve_ivp


def leer_float(mensaje, defecto):
    texto = input(f"{mensaje} [{defecto}]: ").strip()
    return float(texto) if texto else float(defecto)


def leer_int(mensaje, defecto, minimo=1):
    texto = input(f"{mensaje} [{defecto}]: ").strip()
    valor = int(texto) if texto else int(defecto)
    if valor < minimo:
        raise ValueError(f"El valor debe ser mayor o igual que {minimo}.")
    return valor


def imprimir_tabla(x, y, encabezado_x="x", encabezado_y="y", decimales=6):
    print(f"\n{encabezado_x:>14} | {encabezado_y:>16}")
    print("-" * 34)
    for xi, yi in zip(x, y):
        print(f"{xi:14.6f} | {yi:16.{decimales}f}")


# ---------------------------------------------------------------------
# 1. Cable flexible: y'' = -w(x)/T, y(0)=y(L)=0
# ---------------------------------------------------------------------
def problema_cable():
    print("\n--- PROBLEMA 1: FORMA DE UN CABLE ---")
    L = leer_float("Longitud del cable L (m)", 50)
    tension = leer_float("Componente horizontal de la tensión T (kg)", 5000)
    intervalos = leer_int("Número de intervalos (mínimo 10)", 20, 10)

    if L <= 0 or tension <= 0:
        raise ValueError("La longitud y la tensión deben ser positivas.")

    h = L / intervalos
    x = np.linspace(0, L, intervalos + 1)
    w = 20 * (1 + np.exp(x / 25.0))

    # Para nodos interiores:
    # (y[i-1] - 2*y[i] + y[i+1])/h² = -w[i]/T
    n = intervalos - 1
    matriz = np.diag(-2 * np.ones(n))
    matriz += np.diag(np.ones(n - 1), 1)
    matriz += np.diag(np.ones(n - 1), -1)
    rhs = -(h**2) * w[1:-1] / tension
    y_interior = solve(matriz, rhs)

    y = np.zeros(intervalos + 1)
    y[1:-1] = y_interior

    print("\nEcuación utilizada: y''(x) = -w(x)/T")
    print("w(x) = 20[1 + exp(x/25)] kg/m")
    imprimir_tabla(x, y, "x (m)", "y (m)")
    print(f"\nDesplazamiento máximo calculado: {np.max(y):.6f} m")

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, "o-", label="Solución por diferencias finitas")
    plt.xlabel("Distancia x (m)")
    plt.ylabel("Desplazamiento y(x) (m)")
    plt.title("Forma del cable flexible")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 2. Transferencia de calor en un tubo
# -(k A T')' + P hc T = P hc T_inf
# T(0)=100; -k T'(H)=hc(T(H)-T_inf)
# ---------------------------------------------------------------------
def problema_tubo():
    print("\n--- PROBLEMA 2: TEMPERATURA EN UN TUBO ---")
    hc = leer_float("Coeficiente convectivo hc (W/(m² K))", 30)
    H = leer_float("Longitud H (m)", 0.1)
    k = leer_float("Conductividad térmica k (W/(m K))", 100)
    Tinf = leer_float("Temperatura ambiente T_inf (°C)", 20)
    T0 = leer_float("Temperatura en x=0 (°C)", 100)
    intervalos = leer_int("Número de intervalos (mínimo 10)", 20, 10)

    if H <= 0 or k <= 0 or hc < 0:
        raise ValueError("H y k deben ser positivos y hc no negativo.")

    x = np.linspace(0, H, intervalos + 1)
    dx = H / intervalos

    # Funciones dadas en el enunciado.
    A = 0.005 * (0.05 - 0.25 * x)
    P = A / 0.005 + 0.01

    if np.any(A <= 0):
        raise ValueError("El área A(x) debe ser positiva en todo el dominio.")

    # Sistema lineal para los nodos 0...N.
    M = np.zeros((intervalos + 1, intervalos + 1))
    b = np.zeros(intervalos + 1)

    # Condición de Dirichlet en x=0.
    M[0, 0] = 1.0
    b[0] = T0

    # Nodos interiores: discretización conservativa de -(k A T')'.
    for i in range(1, intervalos):
        A_izq = 0.5 * (A[i - 1] + A[i])
        A_der = 0.5 * (A[i] + A[i + 1])
        coef_izq = k * A_izq / dx**2
        coef_der = k * A_der / dx**2

        M[i, i - 1] = -coef_izq
        M[i, i] = coef_izq + coef_der + P[i] * hc
        M[i, i + 1] = -coef_der
        b[i] = P[i] * hc * Tinf

    # Condición de Robin en x=H:
    # -k*(T_N - T_(N-1))/dx = hc*(T_N - T_inf)
    M[-1, -2] = k / dx
    M[-1, -1] = -(k / dx + hc)
    b[-1] = -hc * Tinf

    temperatura = solve(M, b)
    imprimir_tabla(x, temperatura, "x (m)", "T (°C)")
    print(f"\nTemperatura calculada en x=H: {temperatura[-1]:.6f} °C")

    plt.figure(figsize=(9, 5))
    plt.plot(x, temperatura, "o-", label="Diferencias finitas")
    plt.xlabel("Posición x (m)")
    plt.ylabel("Temperatura (°C)")
    plt.title("Distribución de temperatura en el tubo")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 3. Condición -phi'(0) + f_I phi(0) = g_I equivalente a phi(0)=2
# ---------------------------------------------------------------------
def problema_frontera():
    print("\n--- PROBLEMA 3: CONDICIÓN DE FRONTERA ---")
    print("La condición general es: -phi'(0) + f_I*phi(0) = g_I.")
    print("Para imponer phi(0)=2, la derivada no debe intervenir en la")
    print("restricción; se toma f_I distinto de cero y g_I = 2*f_I.")
    fI = leer_float("Elige f_I (distinto de cero)", 1.0e10)

    if fI == 0:
        print("f_I no puede ser cero si se quiere imponer phi(0)=2 mediante")
        print("esta condición. Ejecuta de nuevo y elige otro valor.")
        return

    gI = 2 * fI
    print("\nRespuesta:")
    print(f"f_I = {fI:.6g}")
    print(f"g_I = {gI:.6g}")
    print(f"Relación: g_I/f_I = {gI/fI:.6g} = 2")
    print("Importante: la equivalencia exacta requiere que el término")
    print("-phi'(0) no forme parte de la condición impuesta, o que la")
    print("formulación numérica trate esta ecuación como una penalización.")
    print("Con la ecuación escrita literalmente, -phi'(0)+f_I*phi(0)=g_I")
    print("no impone por sí sola phi(0)=2 si phi'(0) es desconocida.")


# ---------------------------------------------------------------------
# 4. Oscilador no lineal: m*x'' + c*x' + k*x + kprima*x³ = 0
# Unidades coherentes con el enunciado: N, cm, s, kg.
# ---------------------------------------------------------------------
def problema_oscilador():
    print("\n--- PROBLEMA 4: OSCILADOR NO LINEAL ---")
    m = leer_float("Masa m (kg)", 0.01)
    c = leer_float("Amortiguamiento c (N s/cm)", 0.15)
    k = leer_float("Constante lineal k (N/cm)", 2.0)
    kp = leer_float("Constante cúbica k' (N/cm³)", 0.2)
    x0 = leer_float("Desplazamiento inicial x(0) (cm)", 10)
    v0 = leer_float("Velocidad inicial x'(0) (cm/s)", 0)
    t_final = leer_float("Tiempo final (s)", 10)
    puntos = leer_int("Número de puntos de salida", 1000, 2)

    if m <= 0 or t_final <= 0:
        raise ValueError("La masa y el tiempo final deben ser positivos.")

    # Convertimos fuerzas a N y desplazamiento a metros para que las
    # unidades sean consistentes con m en kg:
    # k N/cm -> 100*k N/m; k' N/cm³ -> 1e6*k' N/m³;
    # c N*s/cm -> 100*c N*s/m.
    c_si = 100 * c
    k_si = 100 * k
    kp_si = 1.0e6 * kp
    x0_si = x0 / 100
    v0_si = v0 / 100

    def sistema(t, z):
        pos, vel = z
        aceleracion = -(c_si * vel + k_si * pos + kp_si * pos**3) / m
        return [vel, aceleracion]

    tiempos = np.linspace(0, t_final, puntos)
    solucion = solve_ivp(
        sistema, (0, t_final), [x0_si, v0_si],
        t_eval=tiempos, method="RK45", rtol=1e-8, atol=1e-10
    )
    if not solucion.success:
        raise RuntimeError(solucion.message)

    desplazamiento_cm = solucion.y[0] * 100
    velocidad_cms = solucion.y[1] * 100

    print("\nSe resolvió el sistema equivalente de primer orden con RK45,")
    print("un método adaptativo de Runge-Kutta.")
    print(f"Desplazamiento final: {desplazamiento_cm[-1]:.30f} cm")
    print(f"Velocidad final: {velocidad_cms[-1]:.30f} cm/s")

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(tiempos, desplazamiento_cm)
    ax[0].set_ylabel("Desplazamiento (cm)")
    ax[0].set_title("Movimiento del sistema masa-resorte-amortiguador")
    ax[0].grid(True)
    ax[1].plot(tiempos, velocidad_cms)
    ax[1].set_xlabel("Tiempo (s)")
    ax[1].set_ylabel("Velocidad (cm/s)")
    ax[1].grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 5. Ecuación del calor 2D:
# T_t = alpha*(T_xx + T_yy), bordes T0, interior inicialmente Ti
# Diferencias finitas explícitas.
# ---------------------------------------------------------------------
def problema_placa():
    print("\n--- PROBLEMA 5: TRANSFERENCIA DE CALOR EN UNA PLACA 2D ---")
    Lx = leer_float("Longitud Lx (m)", 1.0)
    Ly = leer_float("Longitud Ly (m)", 1.0)
    alpha = leer_float("Difusividad térmica alpha (m²/s)", 1.0e-4)
    T0 = leer_float("Temperatura constante de los bordes T0 (°C)", 20)
    Ti = leer_float("Temperatura inicial interior Ti (°C)", 100)
    nx = leer_int("Número de intervalos en x (mínimo 10)", 30, 10)
    ny = leer_int("Número de intervalos en y (mínimo 10)", 30, 10)
    t_final = leer_float("Tiempo final de simulación (s)", 1000)

    if Lx <= 0 or Ly <= 0 or alpha <= 0 or t_final <= 0:
        raise ValueError("Longitudes, alpha y tiempo final deben ser positivos.")

    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    dx = Lx / nx
    dy = Ly / ny

    # Condición de estabilidad para el esquema explícito 2D:
    # alpha*dt*(1/dx² + 1/dy²) <= 1/2.
    dt_max = 0.5 / (alpha * (1 / dx**2 + 1 / dy**2))
    dt = 0.9 * dt_max
    pasos = int(np.ceil(t_final / dt))
    dt = t_final / pasos
    rx = alpha * dt / dx**2
    ry = alpha * dt / dy**2

    T = np.full((ny + 1, nx + 1), Ti, dtype=float)
    T[0, :] = T0
    T[-1, :] = T0
    T[:, 0] = T0
    T[:, -1] = T0

    tiempos_deseados = np.array([0, 0.1, 0.25, 0.5, 1.0]) * t_final
    indices_deseados = sorted(set(int(round(t / dt)) for t in tiempos_deseados))
    instantaneas = {0: T.copy()}

    for n in range(1, pasos + 1):
        anterior = T.copy()
        T[1:-1, 1:-1] = (
            anterior[1:-1, 1:-1]
            + rx * (anterior[1:-1, 2:] - 2 * anterior[1:-1, 1:-1]
                    + anterior[1:-1, :-2])
            + ry * (anterior[2:, 1:-1] - 2 * anterior[1:-1, 1:-1]
                    + anterior[:-2, 1:-1])
        )
        # Reafirmar las condiciones de frontera.
        T[0, :] = T0
        T[-1, :] = T0
        T[:, 0] = T0
        T[:, -1] = T0

        if n in indices_deseados:
            instantaneas[n] = T.copy()

    X, Y = np.meshgrid(x, y)
    indices_grafica = sorted(instantaneas.keys())
    fig = plt.figure(figsize=(14, 9))
    for j, idx in enumerate(indices_grafica, start=1):
        ax = fig.add_subplot(2, 3, j, projection="3d")
        superficie = instantaneas[idx]
        ax.plot_surface(X, Y, superficie, cmap="inferno", edgecolor="none")
        ax.set_title(f"t = {idx * dt:.2f} s")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("T (°C)")
        ax.set_zlim(min(T0, Ti), max(T0, Ti))
    plt.suptitle("Distribución de temperatura en la placa")
    plt.tight_layout()
    plt.show()

    # Mapa de calor del último instante.
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, T, levels=20, cmap="inferno")
    plt.colorbar(label="Temperatura (°C)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"Mapa de temperatura al tiempo t={t_final:g} s")
    plt.tight_layout()
    plt.show()

    print(f"\nPaso temporal utilizado: dt = {dt:.6g} s")
    print(f"Número de pasos: {pasos}")
    print(f"Temperatura interior mínima al final: {np.min(T[1:-1, 1:-1]):.4f} °C")
    print(f"Temperatura interior máxima al final: {np.max(T[1:-1, 1:-1]):.4f} °C")


def main():
    opciones = {
        "1": ("Forma del cable", problema_cable),
        "2": ("Enfriamiento del tubo", problema_tubo),
        "3": ("Condición de frontera", problema_frontera),
        "4": ("Oscilador no lineal", problema_oscilador),
        "5": ("Transferencia de calor en una placa", problema_placa),
    }

    while True:
        print("\n" + "=" * 58)
        print("PROGRAMA DE MÉTODOS NUMÉRICOS")
        print("=" * 58)
        for clave, (nombre, _) in opciones.items():
            print(f"{clave}. {nombre}")
        print("0. Salir")

        opcion = input("\nSelecciona un ejercicio: ").strip()
        if opcion == "0":
            print("Programa finalizado.")
            break
        if opcion not in opciones:
            print("Opción no válida. Intenta nuevamente.")
            continue

        try:
            opciones[opcion][1]()
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as error:
            print(f"\nNo se pudo resolver el ejercicio: {error}")


if __name__ == "__main__":
    main()
