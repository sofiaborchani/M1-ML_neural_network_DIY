import numpy as np


#####################################################################################################
######################################## CLASSES LOSS ###############################################
#####################################################################################################

class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class MSELoss(Loss):
    """
    Classe pour la Mean Square Error.
    """

    def forward(self, y, yhat):
        """
        Permet de calculer la mean square error.
        y : la supervision (taille batch x d) (où batch est le nombre d'exemples)
        yhat : la prédiction (taille batch x d)
        Return : coût (taille batch)
        """
        #assert(y.shape==yhat.shape)
        return np.linalg.norm(y-yhat, axis=1)**2
    
    def backward(self, y, yhat):
        """
        Permet de calculer le gradient de coût par rapport à yhat
        y : la supervision (taille batch x d) (où batch est le nombre d'exemples)
        yhat : la prédiction (taille batch x d)
        Return : gradient de coût (taille batch x d)
        """
        #assert(y.shape==yhat.shape)
        return -2 * (y-yhat)

class CELoss(Loss):
    """ 
    Classe pour le coût cross-entropique.
    """
    def forward (self, y, yhat) :
        """ 
        Permet de calculer le coût cross-entropique
        y : la supervision (taille batch x d) (où batch est le nombre d'exemples)
        yhat : la prédiction (taille batch x d)
        Return : coût (taille batch)
        """
        return 1 - np.sum(y * yhat , axis = 1)

    def backward (self, y, yhat) :
        """
        Permet de calculer le gradient de coût par rapport à yhat
        y : la supervision (taille batch x d) (où batch est le nombre d'exemples)
        yhat : la prédiction (taille batch x d)
        Return : gradient de coût (taille batch x d)
        """
        return yhat - y
    
class CElogSMLoss(Loss):
    """ 
    Classe pour le coût cross-entropique dans le cas d'un SoftMax passé au logarithme
    """
    def forward (self, y, yhat, eps=1e-5) :
        """ 
        Permet de calculer le coût cross-entropique dans le cas d'un SoftMax passé au logarithme
        y : la supervision (taille batch x d) (où batch est le nombre d'exemples)
        yhat : la prédiction (taille batch x d)
        Return : coût (taille batch)
        """
        return - np.sum(y * yhat, axis = 1) + np.log( np.sum(np.exp(yhat), axis=1) + eps)


    def backward (self, y, yhat) :
        """
        Permet de calculer le gradient de coût par rapport à yhat.
        y : la supervision (taille batch x d) (où batch est le nombre d'exemples)
        yhat : la prédiction (taille batch x d)
        Return : gradient de coût (taille batch x d)
        """
        return np.exp(yhat) / np.sum(np.exp(yhat), axis=1).reshape((-1,1)) - y 
    

class BCE(Loss):
    """ 
    Classe pour le coût cross-entropique dans le cas d'un SoftMax passé au logarithme
    """
    def forward (self, y, yhat, eps=1e-5) :
        """ 
        Permet de calculer le coût cross-entropique dans le cas d'un SoftMax passé au logarithme
        y : la supervision (taille batch x d) (où batch est le nombre d'exemples)
        yhat : la prédiction (taille batch x d)
        Return : coût (taille batch)
        """
        return - ( y * np.log(yhat + eps) + (1-y) * np.log(1-yhat + eps))

    def backward (self, y, yhat, eps=1e-5) :
        """
        Permet de calculer le gradient de coût par rapport à yhat.
        y : la supervision (taille batch x d) (où batch est le nombre d'exemples)
        yhat : la prédiction (taille batch x d)
        Return : gradient de coût (taille batch x d)
        """
        
        return ((1 - y) / (1 - yhat + eps)) - (y / yhat + eps)


#####################################################################################################
###################################### CLASSES MODULES ##############################################
#####################################################################################################



class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calculé et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass
    
class Linear(Module):
    """
    Classe pour la couche linéaire.
    """
    def __init__(self,  input, output):
        self._input = input
        self._output = output
        self._parameters = 2 * ( np.random.rand(self._input, self._output) - 0.5 ) #matrice de poids (initialisation aléatoire centrée en 0)
        self._biais = np.random.random((1, self._output)) - 0.5
        self.zero_grad() #on initialise en mettant les gradients à 0

    def forward(self, X):
        """"
        Permet de calculer les sorties du module pour les entrées passées en paramètre. 
        X : matrice des entrées (taille batch x input)
        Return : sorties du module (taille batch x output)
        """
        assert X.shape[1] == self._input
        return np.dot(X, self._parameters) + self._biais
    
    def zero_grad(self):
        """
        Permet de réinitialiser le gradient à 0.
        """
        self._gradient=np.zeros((self._input, self._output))
        self._biais_grad = np.zeros((1, self._output))
    
    def update_parameters(self, gradient_step=0.001):
        """
        Permet de mettre à jour les paramètres du module selon le gradient accumulé jusqu'à son appel avec un pas gradient_step
        gradient_step : pas du gradient
        """
        self._parameters -= gradient_step * self._gradient
        self._biais -= gradient_step * self._biais_grad
    
    def backward_update_gradient(self, input, delta):
        """
        Permet de calculer le gradient du coût par rapport aux paramètres et l’additionner à la variable _gradient en fonction de l’entrée input et des δ de la couche suivante delta.
        input: entrée du module
        delta: delta de la couche suivante
        """
        #assert input.shape[1] == self._input
        #assert delta.shape[1] == self._output
        #assert delta.shape[0] == input.shape[0]
        self._gradient += np.dot(input.T, delta)
        self._biais_grad += np.sum(delta, axis=0)

    def backward_delta(self, input, delta):
        """
        Permet de calculer le gradient du coût par rapport aux entrées en fonction de l’entrée input et des deltas de la couche suivante delta.
        input: entrée du module
        delta: delta de la couche suivante
        Return: delta de la couche actuelle
        """
        #assert input.shape[1] == self._input
        #assert delta.shape[1] == self._output
        return np.dot(delta, self._parameters.T)
    
class TanH(Module):
    """
    Classe pour le module non linéaire permettant d'appliquer une tangente hyperbolique aux entrées.
    """

    def __init__(self):
        """
        Constructeur sans paramètres du module TanH. 
        """
        super().__init__()

    def forward(self, input):
        """"
        Permet de calculer les sorties du module pour les entrées passées en paramètre. 
        X : matrice des entrées (taille batch x input)
        Return : sorties du module (taille batch x output)
        """
        return np.tanh(input)
    
    
    def update_parameters(self, gradient_step=0.001):
        """
        Permet de mettre à jour les paramètres du module selon le gradient accumulé jusqu'à son appel avec un pas gradient_step
        gradient_step : pas du gradient
        """
        pass
    
    def backward_update_gradient(self, input, delta):
        """
        Permet de calculer le gradient du coût par rapport aux paramètres et l’additionner à la variable _gradient en fonction de l’entrée input et des δ de la couche suivante delta.
        input: entrée du module
        delta: delta de la couche suivante
        """
        pass

    def backward_delta(self, input, delta):
        """
        Permet de calculer le gradient du coût par rapport aux entrées en fonction de l’entrée input et des deltas de la couche suivante delta.
        input: entrée du module
        delta: delta de la couche suivante
        Return: delta de la couche actuelle
        """
        # assert input.shape[1] == self._input
        # assert delta.shape[1] == self._output
        return ( 1 - np.tanh(input)**2 ) * delta
    

class Sigmoide(Module):
    """
    Classe pour le module non linéaire permettant d'appliquer une sigmoïde aux entrées. 
    """

    def __init__(self):
        """
        Constructeur sans paramètres du module Sigmoïde. 
        """
        super().__init__()

    def forward(self, input):
        """"
        Permet de calculer les sorties du module pour les entrées passées en paramètre. 
        X : matrice des entrées (taille batch x input)
        Return : sorties du module (taille batch x output)
        """
        return 1 / (1 + np.exp(-input))
    
    
    def update_parameters(self, gradient_step=0.001):
        """
        Permet de mettre à jour les paramètres du module selon le gradient accumulé jusqu'à son appel avec un pas gradient_step
        gradient_step : pas du gradient
        """
        pass
    
    def backward_update_gradient(self, input, delta):
        """
        Permet de calculer le gradient du coût par rapport aux paramètres et l’additionner à la variable _gradient en fonction de l’entrée input et des δ de la couche suivante delta.
        input: entrée du module
        delta: delta de la couche suivante
        """
        pass

    def backward_delta(self, input, delta):
        """
        Permet de calculer le gradient du coût par rapport aux entrées en fonction de l’entrée input et des deltas de la couche suivante delta.
        input: entrée du module
        delta: delta de la couche suivante
        Return: delta de la couche actuelle
        """
        # assert input.shape[1] == self._input
        # assert delta.shape[1] == self._output
        return (np.exp(-input) / ( 1 + np.exp(-input))**2 ) * delta


    
#####################################################################################################
########################################## ENCAPSULAGE ##############################################
#####################################################################################################

class Sequential:
    """
    Classe permettant d'ajouter des modules en série et automatisantt les procédures de forward et backward.
    """
    
    def __init__(self, modules):
        self._modules = modules #liste de modules 

    def forward(self, input): 
        """"
        Permet de calculer les sorties du module pour les entrées passées en paramètre. 
        input : matrice des entrées (taille batch x input)
        Return : sorties du module (taille batch x output)
        """
        liste_forwards = [input]
        for i in range(0, len(self._modules)):
            liste_forwards.append(self._modules[i].forward( liste_forwards[-1]))
       
        return liste_forwards[1:]
    
    def update_parameters(self, gradient_step=1e-3):
        """
        Permet de mettre à jour les paramètres du module selon le gradient accumulé jusqu'à son appel avec un pas gradient_step
        gradient_step : pas du gradient
        """
        for module in self._modules :
            module.update_parameters(gradient_step=gradient_step)
            module.zero_grad()

    def backward_delta(self, liste_forwards, delta):
        """
        Permet de calculer le gradient du coût par rapport aux entrées en fonction de l’entrée input et des deltas de la couche suivante delta.
        input: entrée du module
        delta: delta de la couche suivante
        Return: delta de la couche actuelle
        """
        liste_deltas = [delta]
        for i in range(len(self._modules)-1, 0, -1):
            self._modules[i].backward_update_gradient(liste_forwards[i-1], liste_deltas[-1])
            liste_deltas.append( self._modules[i].backward_delta(liste_forwards[i-1], liste_deltas[-1]))
        
        return liste_deltas
    


class Optim:
    """
    Classe permettant de condenser les 
    """
    
    def __init__(self, net, loss, eps):
        
        self._net = net
        self._eps = eps
        self._loss = loss
        
        
    def step(self, batch_x, batch_y):
        """
        Calcule la sortie du réseau sur batch_x, calcule le coût par rapport aux labels batch_y, exécute la passe backward et met à jour les paramètres du réseau. 
        batch_x : batch d'entrées
        batch_y : batch de labels correspondants 
        Return : le coût par rapport aux labels batch_y
        """
        #Calcule sortie du réseau sur batch_x
        liste_forwards = self._net.forward( batch_x )

        #Calcule coût
        batch_y_hat = liste_forwards[-1]
        loss = self._loss.forward(batch_y,batch_y_hat)
        
        #Execute la pass backward
        delta = self._loss.backward(batch_y,batch_y_hat)
        list_deltas = self._net.backward_delta(liste_forwards, delta)

        #Mise à jour des paramètres du réseau        
        self._net.update_parameters(self._eps)

        return loss

   
    def SGD(self, datax, datay, batch_size, iter=10):
        """
        Effectue une descente de gradient.
        datax : jeu de données
        datay : labels correspondans
        batch_size : taille de batch
        iter : nombre d'itérations
        Return : loss
        """
        nb_data = len(datax)
        nb_batches = nb_data // batch_size

        #Si le nombre de data n'est pas divisible par le nombre de batch on aura un batch partiellement rempli
        if nb_data % batch_size != 0:
            nb_batches += 1

        liste_loss = []

        for i in range(iter):
            #Mélange les indices
            indices = np.random.permutation(nb_data)
            datax = datax[indices]
            datay = datay[indices]
            liste_batch = []

            #Forme les batches
            liste_batch_x = np.array_split(datax, nb_batches)
            liste_batch_y = np.array_split(datay, nb_batches)

            #Effectue la descente de gradient pour chaque batch
            for j in range(nb_batches):
                batch_x = liste_batch_x[j]
                batch_y = liste_batch_y[j]
                loss = self.step(batch_x, batch_y)
                liste_batch.append(np.array(loss.mean()))
            liste_loss.append(np.array(liste_batch).mean())

        return liste_loss
    
#####################################################################################################
########################################## MULTI-CLASSE #############################################
#####################################################################################################

class Softmax(Module):

    def __init__(self):
        """
        Constructeur sans paramètres du module Sigmoïde. 
        """
        super().__init__()


    def forward(self, datax):
        """
        Calcule la passe forward en appliquant la fonction Softmax aux entrées
        datax : Entrée de la couche précédente, de taille (batch_size, n_in)
        Return : sortie de la couche (taille (batch_size, n_out))
        """
        exp_x = np.exp(datax - np.max(datax, axis=1)) #soustrait le maximum de chaque ligne pour éviter les débordements
        return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))
        # ou 
        # exp_x = np.exp(datax)
        # return exp_x /np.sum(exp_x, axis=1).reshape(-1,1)

    def update_parameters(self, gradient_step=0.001):
        """
        Permet de mettre à jour les paramètres du module selon le gradient accumulé jusqu'à son appel avec un pas gradient_step
        gradient_step : pas du gradient
        """
        pass
    
    def backward_update_gradient(self, input, delta):
        """
        Permet de calculer le gradient du coût par rapport aux paramètres et l’additionner à la variable _gradient en fonction de l’entrée input et des δ de la couche suivante delta.
        input: entrée du module
        delta: delta de la couche suivante
        """
        pass

    def backward_delta(self, input, delta):
        """
        Calcule la dérivée de l'erreur de sortie par rapport à l'entrée de la couche
        input : entrée de la couche précédente 
        delta : dérivée de l'erreur de sortie par rapport à la sortie de la couche 
        Return : dérivée de l'erreur de sortie par rapport à l'entrée de la couche
        """
        d = self.forward(np.array(input))
        return delta * d * (1-d) 
    
    

#####################################################################################################
########################################## AUTO-ENCODEUR ############################################
#####################################################################################################


class AutoEncodeur:
    
    def __init__(self,encodeur,decodeur):
        """
        Constructeur de la classe AutoEncodeur
        encodeur : prend en entrée une donnée et la transforme dan sun espace de plus petite dimension 
        decodeur : prend en entrée le code d'un exemple et décode l’exemple vers sa représentation initiale
        """
        self.modules = Sequential(encodeur + decodeur)

    def forward(self, datax):
        """
        Calcule la passe forward en appliquant la fonction Softmax aux entrées
        datax : Entrée de la couche précédente, de taille (batch_size, n_in)
        Return : sortie de la couche (taille (batch_size, n_out))
        """
        return self.modules.forward(datax)
                               
    def update_parameters(self, gradient_step=0.001):
        """
        Permet de mettre à jour les paramètres du module selon le gradient accumulé jusqu'à son appel avec un pas gradient_step
        gradient_step : pas du gradient
        """
        return self.modules.update_parameters(gradient_step)
    
    def backward_update_gradient(self, input, delta):
        """
        Permet de calculer le gradient du coût par rapport aux paramètres et l’additionner à la variable _gradient en fonction de l’entrée input et des δ de la couche suivante delta.
        input: entrée du module
        delta: delta de la couche suivante
        """
        return self.modules.backward_update_gradient(input, delta)

    def backward_delta(self, input, delta):
        """
        Calcule la dérivée de l'erreur de sortie par rapport à l'entrée de la couche
        input : entrée de la couche précédente 
        delta : dérivée de l'erreur de sortie par rapport à la sortie de la couche 
        Return : dérivée de l'erreur de sortie par rapport à l'entrée de la couche
        """
        return self.modules.backward_delta(input, delta) 


#####################################################################################################
########################################## CONVOLUTION ##############################################
#####################################################################################################

# PARTIE 6 : Convolution

class ReLU(Module):

    def __init__(self):
        """
        Constructeur de la classe ReLU
        """
        super().__init__()


    def forward(self, datax):
        """
        Calcule la passe forward 
        datax : Entrée de la couche précédente, de taille (batch_size, n_in)
        Return : sortie de la couche 
        """
        return np.maximum(0,datax)


    def update_parameters(self, gradient_step=0.001):
        pass
    
    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        """
        Calcule la dérivée de l'erreur de sortie par rapport à l'entrée de la couche
        input : entrée de la couche précédente 
        delta : dérivée de l'erreur de sortie par rapport à la sortie de la couche 
        Return : dérivée de l'erreur de sortie par rapport à l'entrée de la couche
        """
        return np.where(input<0, 0, 1) * delta


    
    
class Flatten(Module):
    def __init__(self):
            """
            Constructeur de la classe Flatten
            """
            super().__init__() 

    def forward(self, datax):
        """
        Calcule la passe forward 
        datax : Entrée de la couche précédente, de taille 2D d_out * C
        Return : sortie de la couche taille d_out * C
        """
        return datax.reshape(datax.shape[0], -1)


    def update_parameters(self, gradient_step=0.001):
        pass
    
    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        """
        Calcule la dérivée de l'erreur de sortie par rapport à l'entrée de la couche
        input : entrée de la couche précédente 
        delta : dérivée de l'erreur de sortie par rapport à la sortie de la couche 
        Return : dérivée de l'erreur de sortie par rapport à l'entrée de la couche
        """
        return delta.reshape(input.shape)

class Conv1D(Module):

    def __init__(self, k_size, chan_in, chan_out, stride):
        """
        Constructeur de la classe Conv1D
        """
        self._k_size = k_size
        self._chan_in = chan_in
        self._chan_out = chan_out
        self._stride = stride 

        #self._parameters = 2 * ( np.random.rand(k_size, chan_in) - 0.5 )
        self._parameters = np.random.randn(k_size, chan_in, chan_out)
        self._gradient = np.zeros(self._parameters.shape)
        self._biais = np.zeros((chan_out)) 
        
        #self.params = np.random.rand(k_size, chan_in, chan_out) - 0.5
        #self.bias = np.zeros((chan_out,))

        
    def zero_grad(self):
        """
        Permet de réinitialiser le gradient à 0.
        """
        self._gradient=np.zeros(self._gradient.shape)
        self._biais = np.zeros(self._biais.shape)
    
    
    def forward(self, input):
        """
        Calcule la sortie de la couche Conv1D
        input: un batch d'entrée de taille (batch, length, chan_in)
        Return : sortie de la couche de taille (batch, (length - k_size) // stride + 1, chan_out)
        """
        batch_size, length, chan_in = input.shape
        
        d_out_size = (length - self._k_size) // self._stride + 1
        o = np.zeros((batch_size, d_out_size, self._chan_out))
        
        for i in range(0, d_out_size, self._stride):
            batch = input[:, i:i+self._k_size, :]
            oi = np.tensordot(batch, self._parameters, axes=((1,2),(0,1))) 
            o[:, i, :] = oi
        
        return o + self._biais.reshape((1, 1, self._chan_out))
    
                               
    def update_parameters(self, gradient_step=0.001):
        """
        Permet de mettre à jour les paramètres du module selon le gradient accumulé jusqu'à son appel avec un pas gradient_step
        gradient_step : pas du gradient
        """
        self._parameters -= gradient_step * self._gradient
        self._biais -= gradient_step * self._biais
    
    def backward_update_gradient(self, input, delta):
        """
        Permet de calculer le gradient du coût par rapport aux paramètres et l’additionner à la variable _gradient en fonction de l’entrée input et des δ de la couche suivante delta.
        input: entrée du module
        delta: delta de la couche suivante
        """
        d_out_size = ( (input.shape[0]-self._k_size) // self._stride ) + 1
    
        o = np.zeros((delta.shape[1], d_out_size, input.shape[2], self._chan_out)) 

        for i in range(0, d_out_size, self._stride):
            oi = delta[:,i,:].T @ input[:, i:i+self._k_size, :].reshape(input.shape[0], -1)
            o[:,i,:] = oi

        self._gradient = ( np.sum(o, axis=0).T / delta.shape[0]).reshape(self._gradient.shape)
        self._biais = delta.mean((0,1))

    def backward_delta(self, input, delta):
        """
        Calcule la dérivée de l'erreur de sortie par rapport à l'entrée de la couche
        input : entrée de la couche précédente 
        delta : dérivée de l'erreur de sortie par rapport à la sortie de la couche 
        Return : dérivée de l'erreur de sortie par rapport à l'entrée de la couche
        """
        d_out_size = ( (input.shape[1] - self._k_size) // self._stride ) + 1
        
        o = np.zeros(input.shape)

        for i in range(0, d_out_size, self._stride) : 
            o[:, i:i+self._k_size, :] += (( delta[:,i,:]) @ ( self._parameters.reshape(-1, self.chan_out).T)).reshape(input.shape[0],self.k_size,self.chan_in)

        return o
    






class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        """
        Constructeur de la classe Conv1D
        """
        super().__init__()
        self._k_size = k_size
        self._stride = stride 

    
    def forward(self, input):
        """
        Calcule la sortie de la couche MaxPool1D
        input: un batch d'entrée de taille (batch, length, chan_in)
        Return : sortie de la couche de taille (batch, (length - k_size) // stride + 1, chan_in)
        """
        batch_size, length, chan_in = input.shape
        
        d_out_size = (length - self._k_size) // self._stride + 1
        o = np.zeros((batch_size, d_out_size, chan_in))

        for i in range(0,d_out_size,self._stride):
                
            batch = input[:, i:i+self._stride, :]
            oi = np.max(batch, axis=1)
            o[:, i, :] = oi
        
        return o
 
    
    def update_parameters(self, gradient_step=0.001):
        pass
    
    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        """
        Calcule la dérivée de l'erreur de sortie par rapport à l'entrée de la couche
        input : entrée de la couche précédente 
        delta : dérivée de l'erreur de sortie par rapport à la sortie de la couche 
        Return : dérivée de l'erreur de sortie par rapport à l'entrée de la couche
        """
        d_out_size = ( (input.shape[1] - self._k_size) // self._stride ) + 1
        
        o = np.zeros(input.shape)

        for i in range(0, d_out_size, self._stride) : 
            indices_max = (np.argmax(input[:, i:i+self._k_size,:], axis=1) + i).flatten()
            indices_batch = np.repeat(np.arange(input.shape[0]), input.shape[2])
            indices_chan = np.tile(np.arange(input.shape[2]), input.shape[0])

            o[indices_batch, indices_max, indices_chan] = delta[:, i, :].flatten()

        return o
    
    
    
class AvgPool1D(Module):
    def __init__(self, k_size, stride):
        """
        Constructeur de la classe Conv1D
        """
        super().__init__()
        self._k_size = k_size
        self._stride = stride 

    
    def forward(self, input):
        """
        Calcule la sortie de la couche MaxPool1D
        input: un batch d'entrée de taille (batch, length, chan_in)
        Return : sortie de la couche de taille (batch, (length - k_size) // stride + 1, chan_in)
        """
        batch_size, length, chan_in = input.shape
        
        d_out_size = (length - self._k_size) // self._stride + 1
        o = np.zeros((batch_size, d_out_size, chan_in))

        for i in range(0,d_out_size,self._stride):
                
            batch = input[:, i:i+self._stride, :]
            oi = np.mean(batch, axis=1)
            o[:, i, :] = oi
        
        return o
 
    
    def update_parameters(self, gradient_step=0.001):
        pass
    
    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        """
        Calcule la dérivée de l'erreur de sortie par rapport à l'entrée de la couche
        input : entrée de la couche précédente 
        delta : dérivée de l'erreur de sortie par rapport à la sortie de la couche 
        Return : dérivée de l'erreur de sortie par rapport à l'entrée de la couche
        """
        d_out_size = ( (input.shape[1] - self._k_size) // self._stride ) + 1
        
        o = np.zeros(input.shape)

        for i in range(0, d_out_size, self._stride) : 
            indices_batch = np.repeat(np.arange(input.shape[0]), input.shape[2])
            indices_chan = np.tile(np.arange(input.shape[2]), input.shape[0])

            o[:, i:i+self._k_size, :] += np.expand_dims(delta[:, i, :], axis=1) / self._k_size
            
        return o