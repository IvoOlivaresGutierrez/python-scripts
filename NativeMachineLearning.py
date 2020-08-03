# Ejemplo de MachineLearning sin Framework

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

# Crear el Dataset

# numero de registros en los datos
n = 500
# Cuantas caracteristicas tenemos sobre nuestros datos
p = 2

# Dibujo de dos circulos, factor indica la distancia entre los circulos
# Noise sirve para añadir ruido a los datos, para comporbar lo dicho borrar el argumento noise=x
X, Y = make_circles(n_samples=n, factor=0.5, noise = 0.05)
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c="skyblue")
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c="salmon")
plt.axis("equal")
plt.show()

# CLASE DE LA CAPA DE LA RED
class neural_layer():

	def __init__(self, n_conn, n_neur, act_f):
		# Funcion de activacion
		self.act_f = act_f

		# Parametro de ballas, tendremos tantos como neuronas tenga la capa
		# Inicializacion de la capa de forma normalizada y estandarizada
		self.b = np.random.rand(1, n_neur) * 2 - 1
		self.b = np.random.rand(n_conn, n_neur) * 2 - 1

# FUNCIONES DE ACTIVACION
# Funcion por la cual pasa la suma ponderada que se realiza en la neurona
# introduce dentro de la red neuronal no linealidades, permite integrar muchas neuronas

# Existen diferentes tipos de  funcion de activacion como sigmoide, ReLu, tangencial hiperbolica, etc
# en este caso se utilizara la sigmoide

sigm = (lambda x:1 / (1 +np.e ** (-x)),
		lambda x: x * (1 - x))

# Ejemplo de funcion ReLu
# relu = lambda x: np.maximum(0, x)

# variable de -5 a 5 con 100 valores
_x =np.linspace(-5, 5, 100)

# Eje x y eje y luego de pasar por la funcion sigmoide
# Poder visualizar las coordenadas en un plano 2d
plt.plot(_x, sigm[0](_x))
plt.show()

# Ejemplo de instanciacion de capa de red neuoranl layer 0
# l0 = neural_layer(p, 4, sigm)

# Define el numero de neuroans que tiene la red P = numero de neuronas de la primera capa

# Crea la red neuronal
def create_nn(topology, act_f):

	# Verctor neural network
	nn = []

	# Recorre el vector topology hasta el ultimo valor (evita overflow mediante [:-1] descartando el ultimo valor)
	for l, layer in enumerate(topology[:-1]):
		nn.append(neural_layer(topology[l], topology[l+1], act_f))

	return nn


# entrenamiento de la red neuronal, tiene 3 fases
# muestras un dato de entrada e indicas cual debe ser la salida
# una vez teniendo el resultado se compara con el vector real teniendo en cuenta una funcion de coste
# eso genera un error y se debe utilizar para el backpropagation para calcular las derivadaas parciales 
# y posteriormente calcular el descenso del gradiente para optimizar la funcion de coste y entrenar la red

topology = [p, 4 ,8 ,16 ,8, 4, 1]

# Verificar la creacion de la red neuronal
# print(create_nn(topology, sigm))
neural_net = create_nn(topology, sigm)

l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
		   lambda Yp, Yr: (Yp - Yr))

# lr = learning rate es un factor que multiplica el vector gradiente en el descenso del gradiente
# permite identificar en que grado se actualizan los parametros en base a la informacion lr
# lr = velocidad con la que la red neuronal aprende, se debe estudiar a fondo para cada caso
def train(neural_net, X, Y, l2_cost, lr=0.5):

	# Fordward pass
	# Toma el vector de entrada, lo pasa capa por capa ejecutando cada una de las operaciones que se realizan en las neuronas
	# suma ponderada a funcion de activacion y visceversa

	z = X @ neural_net[0].W +neural_net[0].b