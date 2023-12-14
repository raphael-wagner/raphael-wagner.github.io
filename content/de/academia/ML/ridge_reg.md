```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

np.random.seed(42)

num_samples = 100

# generate the independent variable (x) as a random sample from a uniform distribution
X = np.random.uniform(low=0.0, high=6, size=num_samples)

# generate the dependent variable (y) as sin(x) with some gaussian noise
noise = np.random.normal(scale=0.25, size=num_samples)
y = np.sin(X).ravel() + noise

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# plot sample data
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Nonlinear sample data')
plt.show()   
```


    
![png](ridge_reg_files/ridge_reg_0_0.png)
    



```python
# Fit a ridge regression model with gaussian kernel
# Use grid-search cross-validation to find good parameter combinations alpha (regularization) and gamma = 1/sigma

kr_cv = GridSearchCV(
    KernelRidge(kernel="rbf", gamma=0.1),    
    param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.linspace(1, 100, 10)},
)
kr_cv.fit(X, y)

X_plot = np.linspace(0, 6, 1000)[:, None]

plt.scatter(X, y, c='k', label="training data")
plt.plot(X_plot, kr_cv.predict(X_plot), c='r', label="y = predicted_labels(x)")
plt.plot(X_plot, np.sin(X_plot).ravel(), c='b', label="y = sin(x) (''true'' labels)")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Kernel ridge regression")
plt.legend()
```




    <matplotlib.legend.Legend at 0x238f07d75e0>




    
![png](ridge_reg_files/ridge_reg_1_1.png)
    

