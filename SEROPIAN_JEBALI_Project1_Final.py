import matplotlib.pyplot as plt
import numpy as np
import pylops 
import scipy.sparse.linalg
import time
import os
  
def load_image_option_I(bz=0.1, bx=0.3):
    sampling = 5
    im = np.load("dog_rgb.npy")[::sampling, ::sampling, 2]
    Nz, Nx = im.shape

    # Blurring Gaussian operator
    nh = [15, 25]
    hz = np.exp(-bz * np.linspace(-(nh[0] // 2), nh[0] // 2, nh[0]) ** 2)
    hx = np.exp(-bx * np.linspace(-(nh[1] // 2), nh[1] // 2, nh[1]) ** 2)
    hz /= np.trapezoid(hz)  # normalize the integral to 1
    hx /= np.trapezoid(hx)  # normalize the integral to 1
    h = hz[:, np.newaxis] * hx[np.newaxis, :]

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    him = ax.imshow(h)
    ax.set_title("Blurring operator")
    fig.colorbar(him, ax=ax)
    ax.axis("tight")
    plt.show()
    Cop = pylops.signalprocessing.Convolve2D(
        (Nz, Nx), h=h, offset=(nh[0] // 2, nh[1] // 2), dtype="float32"
    )

    imblur = Cop * im
    plt.imshow(im, cmap="viridis", vmin=0, vmax=255)
    plt.show()
    plt.imshow(imblur, cmap="viridis", vmin=0, vmax=255)
    plt.show()

    Wop = pylops.signalprocessing.DWT2D((Nz, Nx), wavelet="haar", level=3)

    # This is your A and b for your f1 cost!
    A = Cop * Wop.H
    b = imblur.ravel()

    return Wop, A, b, im, imblur

def load_image_option_II(bz=0.1, bx=0.3):
    sampling = 2
    im = np.load("chateau.npy")[::sampling, ::sampling, 1]
    Nz, Nx = im.shape

    # Blurring Gaussian operator
    nh = [15, 25]
    hz = np.exp(bz * np.linspace(-(nh[0] // 2), nh[0] // 2, nh[0]) ** 2)
    hx = np.exp(bx * np.linspace(-(nh[1] // 2), nh[1] // 2, nh[1]) ** 2)
    hz /= np.trapezoid(hz)  # normalize the integral to 1
    hx /= np.trapezoid(hx)  # normalize the integral to 1
    h = hz[:, np.newaxis] * hx[np.newaxis, :]

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    him = ax.imshow(h)
    ax.set_title("Blurring operator")
    fig.colorbar(him, ax=ax)
    ax.axis("tight")
    plt.show()
    Cop = pylops.signalprocessing.Convolve2D(
        (Nz, Nx), h=h, offset=(nh[0] // 2, nh[1] // 2), dtype="float32"
    )

    imblur = Cop * im
    plt.imshow(im, cmap="gray", vmin=0, vmax=255)
    plt.show()
    plt.imshow(imblur, cmap="gray", vmin=0, vmax=255)
    plt.show()

    Wop = pylops.signalprocessing.DWT2D((Nz, Nx), wavelet="haar", level=3)

    # This is your A and b for your f1 cost!
    A = Cop * Wop.H
    b = imblur.ravel()

    return Wop, A, b, im, imblur


def my_fista(A, b, opt_cost, eps=10**(-1), niter=100, tol=1e-10, acceleration=False, p=1):
    """ Here you can code your ISTA and FISTA algorithm
        Return: optimal x, and opt_gap_cost (history of cost-optcost)
    """
    
    start_time = time.time()

    L = np.abs((A.T@A).eigs(neigs = 1, symmetric = True)[0])
    alpha = 1.0 / (p*L)

    print(f"\nRunning {'FISTA' if acceleration else 'ISTA'}... alpha = {alpha}")

    x = np.zeros(A.shape[1])
    
    y = np.copy(x) # y_k
    l = 0.0        # lambda_k

    cost = np.zeros(niter + 1)    
    cost[0] = 0.5 * (np.linalg.norm(A @ x - b)**2) + eps * np.linalg.norm(x, 1)

    for k in range(niter):
        
        if acceleration:
            if k > 0:
                gamma = 2*(1-l)/(1+np.sqrt(1+4*l**2))
            else:
                gamma = 0
            pass
        
        grad_f = A.T @ (A @ x - b)
        v = x - alpha * grad_f
        
        threshold = alpha*eps
        
        # this lign replaces the 'for' loop
        y_next = np.sign(v) * np.maximum(np.abs(v) - threshold, 0)
        
        if acceleration:
            l_next = (1 + np.sqrt(1 + 4 * l**2)) / 2
            gamma = (1 - l) / l_next
            
            x_next = gamma * y + (1 - gamma) * y_next
            
            x = x_next
            y = y_next
            l = l_next
        else:
            x = y_next
            
        cost[k+1] = 0.5 * (np.linalg.norm(A @ x - b)**2) + eps * np.linalg.norm(x, 1)
        
        if np.linalg.norm(grad_f) < tol:
            print(f"Converged at iteration {k}")
            break

    cost = cost[:k+2] # we cut the zeros at the end of the cost array for a cleaner plot
    
    # baseline comparaison
    opt_gap_cost = cost - opt_cost

    end_time = time.time()

    algo_name = "FISTA" if acceleration else "ISTA"
    print(f"{algo_name} execution time: {end_time - start_time:.4f} seconds")

    return x, opt_gap_cost

def douglas_rachford_alg(A, b, opt_cost, eps=10**(-1), niter=100, tol=1e-5, maxiter = 100):

    start_time = time.time()

    print("\nRunning Douglas-Rachford...")

    z = np.zeros(A.shape[1])
    x = np.zeros(A.shape[1])

    Op = (A.T @ A) + pylops.Identity(A.shape[1])

    cost = np.zeros(niter + 1)
    cost[0] = 0.5*(np.linalg.norm(A@x - b)**2) + eps*np.linalg.norm(x, 1)

    # we keep a guess for the solution of the linear system to speed up the convergence of the cg solver
    y_guess = np.zeros(A.shape[1])
    for k in range(niter):
        x = np.sign(z)*np.maximum(np.abs(z) - eps, 0)

        rhs = A.T@b + (2*x - z)        
        y, info = scipy.sparse.linalg.cg(Op, rhs, x0=y_guess, rtol=1e-10, maxiter=maxiter)
        y_guess = y # update the guess for the next iteration
        z = z + y - x
        cost[k+1] = 0.5 * (np.linalg.norm(A @ x - b)**2) + eps * np.linalg.norm(x, 1)

    # baseline comparaison
    opt_gap_cost = cost - opt_cost

    end_time = time.time()
    print(f"Douglas-Rachford execution time: {end_time - start_time:.4f} seconds")

    return x, opt_gap_cost

def run_program(A, b, Wop, eps_value=0.1, baseline_iter=1000, tol=1e-10, my_iter=100, maxiter_DR=1000, p=1, bz=0.1, bx=0.3):
    
    print(f"\n  -> Params: eps={eps_value}, p={p}, blur=({bz}, {bx})")

    filename = f"baseline_opt_costs/opt_cost_baseline_eps{eps_value:.0e}_iter{baseline_iter}_bz{bz}_bx{bx}.npy"    
    if os.path.exists(filename):
        opt_cost = np.load(filename)
    else:
        print("     Calculating baseline...")
        imdeblurfista0, n_eff_iter, cost_history = pylops.optimization.sparsity.fista(
            A, b, eps=eps_value, niter=baseline_iter)
        opt_cost = cost_history[-1]
        
        np.save(filename, opt_cost)

    # ISTA
    my_imdeblurfista, opt_gap_cost = my_fista(
        A, b, opt_cost, eps=eps_value, niter=my_iter, tol=tol, acceleration=False, p=p)

    # FISTA
    my_imdeblurfista1, opt_gap_cost1 = my_fista(
        A, b, opt_cost, eps=eps_value, niter=my_iter, tol=tol, acceleration=True, p=p)	

    # Douglas-Rachford
    my_imdeblurfista2, opt_gap_cost2 = douglas_rachford_alg(
        A, b, opt_cost, eps=eps_value, niter=my_iter, tol=tol, maxiter=maxiter_DR)

    plt.figure(figsize=(10, 6))
    plt.loglog(opt_gap_cost, 'C0', label='ISTA')
    plt.loglog(opt_gap_cost1, 'C1', label='FISTA')
    plt.loglog(opt_gap_cost2, 'C2', label='Douglas Rachford')

    plt.grid(True, which="both", ls="-")
    plt.loglog([3, 30], [1e6, 1e5], 'C0--', label='1/k')
    plt.loglog([3, 30], [.5e5, .5e3], 'C1--', label='1/k2')

    plt.xlabel("Number of iterations")
    plt.ylabel("Optimality gap: F - F*")
    plt.title(f"Convergence: eps={eps_value}, alpha=1/({p}*L), blur=({bz},{bx})")

    plt.legend()
    plt.show()

    imdeblurfista = my_imdeblurfista1.reshape(A.dims)
    imdeblurfista = Wop.H * imdeblurfista

    imdeblurDR = my_imdeblurfista2.reshape(A.dims)
    imdeblurDR = Wop.H * imdeblurDR

    return imdeblurfista, imdeblurDR

def visualise_results(im, imblur, imdeblurfista, imdeblurDR):
    #Change viridis into gray for castle image.

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Deblurring", fontsize=14, fontweight="bold", y=0.95)
    ax1 = plt.subplot2grid((2, 5), (0, 0))
    ax2 = plt.subplot2grid((2, 5), (0, 1))
    ax3 = plt.subplot2grid((2, 5), (0, 2))
    ax4 = plt.subplot2grid((2, 5), (0, 3))

    ax1.imshow(im, cmap="viridis", vmin=0, vmax=250)
    ax1.axis("tight")
    ax1.set_title("Original")
    ax2.imshow(imblur, cmap="viridis", vmin=0, vmax=250)
    ax2.axis("tight")
    ax2.set_title("Blurred")

    ax3.imshow(imdeblurfista, cmap="viridis", vmin=0, vmax=250)
    ax3.axis("tight")
    ax3.set_title("FISTA deblurred")

    ax4.imshow(imdeblurDR, cmap="viridis", vmin=0, vmax=250)
    ax4.axis("tight")
    ax4.set_title("Douglas-Rachford deblurred")

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)

    plt.show()


## Load the image according to your option : I for the dog, II for the castle
bz = 0.1 # blurring parameter in the vertical direction (default is 0.1)
bx = 0.3 # blurring parameter in the horizontal direction (default is 0.3)
Wop, A, b, im, imblur = load_image_option_I(bz, bx)

## Run program you have coded:
eps_value = 1e-1
baseline_iter = 5000
tol = 1e-20
my_iter = 5000
maxiter_DR = 50
p=1

start_global = time.time()

imdeblurfista, imdeblurDR = run_program(A, b, Wop, eps_value, baseline_iter, tol, my_iter, maxiter_DR, p)

end_global = time.time()

print(f"\nDone in {end_global - start_global:.2f} seconds.\n")

## Visualise your image results
visualise_results(im, imblur, imdeblurfista, imdeblurDR)
