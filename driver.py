from data.dataset_generator import Generator, number_to_index
from nn.network import Network

numbers = [6, 5, 2]
layers = [784, 100, 3]
net = Network(layers, number_to_index(numbers))

# cria um dataset de 60000 imagens, sendo 50000 para treino e 10000 para teste.
# os números são limitados a 6, 5, 2, que são os correspondentes as 3 iniciais:
# generator = Generator(50000, 10000, numbers, 28)

# cria o dataset com 3 imagens de teste correspondentes as 3 iniciais, em numeros
generator = Generator(0, 3, numbers, 28, True)
dataset = generator.gen()

# para treinar:
# net.grad_descent(dataset[:50000], 200, 10, 3.0, test_data=dataset[50000:])

# carrega uma rede já treinada anteriormente:
net.load('pre-trained-data/biases.npy', 'pre-trained-data/weights.npy')

# porcentagem de acerto:
print((net.evaluate(dataset) / 3) * 100)
