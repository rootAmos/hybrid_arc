
### Hybrid Propulsion System Model

**Overview**
- Four propeller pods
- Two electric motors and one gas turbine per pod, connected mechanically via gearbox
- Gas turbine capable of charging the batteries via the electric motor

**Key Inputs**
- *Pod Shaft Power Required* ($P_{\mathrm{pod}}$)
- *GT Throttle Ratio* ($\lambda_{\mathrm{GT,pod}}$): ratio between the power supplied by the gas turbine and the power required by the pod

$$
\lambda_{\mathrm{GT,pod}} = \frac{P_{\mathrm{gt}}}{P_{\mathrm{pod}}}
$$

Where:
- $\lambda_{\mathrm{GT,pod}} = 1$: Conventional mode (GT power only)
- $\lambda_{\mathrm{GT,pod}} < 1$: Hybrid mode (GT and E-motor power)
- $\lambda_{\mathrm{GT,pod}} > 1$: Generation mode (GT charges batteries via E-motor)

**Key Outputs**
$$\begin{align*}
P_{\mathrm{gt}} &= \lambda_{\mathrm{GT,pod}} \times P_{\mathrm{pod}} \\
\lambda_{\mathrm{GT}} &= \frac{P_{\mathrm{gt}}}{P_{\mathrm{gt,max}}} \\
P_{\mathrm{e}} &= P_{\mathrm{pod}} - P_{\mathrm{gt}} \\
\lambda_{\mathrm{motor}} &= \frac{P_{\mathrm{e}}}{N_{\mathrm{motors}} \times P_{\mathrm{motor,max}}} \\
\gamma &= \frac{P_{\mathrm{e}}}{P_{\mathrm{pod}}}
\end{align*}
$$

**Where**
- $P_{\mathrm{gt}}$: Gas turbine mechanical power per pod
- $P_{\mathrm{e}}$: Total Electric motor mechanical power per pod
- $N_{\mathrm{motors}}$: Number of electric motors per pod
- $\lambda_{\mathrm{GT}}$: Gas turbine throttle (0 to 1)
- $\lambda_{\mathrm{motor}}$: Electric motor throttle (0 to 1)
- $\gamma$: Hybridization factor (-1 to 1)
  - $\gamma > 0$: Electric motors providing power
  - $\gamma < 0$: Electric motors receiving power (battery charging)
  - $\gamma = 0$: Pure gas turbine operation