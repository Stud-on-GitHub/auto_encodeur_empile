
# AUTO-ENCODEUR EMPLIE



# PART 1 - Préprocessing


# Librairies
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# Importation du jeu de données
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')


# Préparation du jeu d'entrainement et du jeu de test
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')


# Obtenir le nombre d'utilisateurs et le nombre de films
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))


# Conversion des données en matrice/liste de liste
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        #
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        """
        id_movies = data[data[:, 0] == id_users, 1]
        id_ratings = data[data[:, 0] == id_users, 2]
        """
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)


# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)



# PART 2 - Créer l'architecture de l'auto-encodeur


class SAE(nn.Module):   
    
    def __init__(self, ):   #fonction d'initialisation        
        super(SAE, self).__init__()
        #étape d'encodage
        self.fc1 = nn.Linear(nb_movies, 20)   #première couche cachée avec 20 node
        self.fc2 = nn.Linear(20, 10)   #deuxième couche cachée à  10 node
        #étape de décodage
        self.fc3 = nn.Linear(10, 20)   #troisième couche cachée à  20 node
        self.fc4 = nn.Linear(20, nb_movies)   #quatrième couche cachée à  20 node
        self.activation = nn.Sigmoid()   #fonction d'activation sigmoide
        
    def forward(self, x):   #fonction d'encodage et de décodage       
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
sae = SAE()

# Critère de cacule de la fonction de cout RMSE racine de la moyen des erreurs au carré
criterion = nn.MSELoss()

# Fonction d'optmisation pour le calcule du gradient
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)   #lr= learning rate/taux d'apprentissage, weight-decay=taux de diminution du taux d'apprentissage à  chaque époques



# PART 3 - Entrainement de l'auto-encodeur


nb_epoch = 200

for epoch in range(1, nb_epoch + 1):
    train_loss = 0   #initialisation du cout
    s = 0.   #compteur en float représentant le nb d'utilisateur ayant noté
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)   #convertit en 1 tableau à  2 dimensions à  l'index 0=les lignes
        target = input.clone()   #varible qui permet la comparaison de la sortie avec l'entrée
        if torch.sum(target.data > 0) > 0:   #teste s'il y a au - 1 vraie note pour l'utilisateur
            output = sae(input)   #calcule de la sortie
            target.require_grad = False   #exclu target du calcule de l'algo du gradien
            output[target == 0] = 0   #ignore les cas sans note
            loss = criterion(output, target)   #calcule de l'erreur pour la rétropropagation
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)   #facteur correctif prenant en compte que tout les utilisateurs ne notent pas tout les films
            #application de l'algo du gradien /calcule
            loss.backward()   #détermine la direction de la mise à  jour des poids
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.   #incrémente le compteur
            optimizer.step()   #détermine l'intencité de la mise à  jour des poids
            
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))



# PART 4 - Teste de l'auto-encodeur
    
    
test_loss = 0   #initialisation du cout

s = 0.   #compteur en float représentant le nb d'utilisateur ayant noté

for id_user in range(nb_users):
    #on va chercher les données que l'on connait pour faire les prédictions
    input = Variable(training_set[id_user]).unsqueeze(0)   #convertit en 1 tableau à  2 dimensions à  l'index 0=les lignes
    #on va chercher les données de teste pour comparer aux prédictions
    target = Variable(test_set[id_user])   #varible qui permet la comparaison de la sortie avec l'entrée
    if torch.sum(target.data > 0) > 0:   #teste s'il y a au - 1 vraie note pour l'utilisateur
        #on fait les prédictions
        output = sae(input)   #calcule de la sortie
        target.require_grad = False   #exclu target du calcule de l'algo du gradien
        output[target == 0] = 0   #ignore les cas sans note
        loss = criterion(output, target)   #calcule de l'erreur pour la rétropropagation
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)   #facteur correctif prenant en compte que tout les utilisateurs ne notent pas tout les films
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.   #incrémente le compteur
        
print('test loss: '+str(test_loss/s))
