import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
np.random.seed(40)
from sklearn.datasets import load_iris

class K_MEANS ():
    def __init__ (self, data, pso = False, k =3, iterations = 1, n_particles = 1):
        self.data = data
        self. pso = pso
        #number of centroids per particle
        self.k = k
        self.iterations = iterations
        #number of particles
        self.n_particles = n_particles
        
    def initialize_particles (self): 
        n = self.n_particles # dit is dus gewoon 1 een bij de normale k-means variant
        particles = []
        for i in range(n):
            #sample a random value
            index = np.random.choice(len(self.data), self.k, replace=False)
            particles.append(self.data[index])
        return particles
    
    def get_closest_cluster(self, point, particle):
        distances = []
        for centroid in particle:
            distances.append(np.linalg.norm(centroid - point))
        return distances.index(min(distances))
    
    def get_distance_cluster(self, point, particle):
        distances = []
        for centroid in particle:
            distances.append(np.linalg.norm(centroid - point))
        return min(distances)
    
    def get_fitness(self, particle_index, cluster_values, cluster_distances):
        means = []
        for i in range(self.k):
            #j == the cluster in the particle
            data = self.data
            filter_array = np.array(cluster_values[particle_index]) == i
            distances = data[filter_array]
            
            average_distance = np.mean(distances)
            
            means.append(average_distance * len(distances)/len(data))
        return sum(means)
    
    def plot_particle(self, particle_index, cluster_values, particles):
        centroids = pd.DataFrame(particles[particle_index])
        df = pd.DataFrame(data=self.data)
        df['closest'] = cluster_values[particle_index]
       
        sb.scatterplot(data = df, x = 0, y = 1, hue='closest')
        sb.scatterplot(data =  centroids, x = 0, y = 1, color= 'red')
        plt.show()
        
        plt.show()
            
                      
    
    def run(self):
        particles = self.initialize_particles()
  
        cluster_values = []
        cluster_distances = []
        
         #dit is een lijst waarbij per particle bepaald wordt bij welke clusters de data points horen
        for i in range(self.n_particles):
            cluster_value = [self.get_closest_cluster(x, particles[i]) for x in self.data]
            cluster_values.append(cluster_value)
            cluster_distance = [self.get_distance_cluster(x, particles[i]) for x in self.data]
            cluster_distances.append(cluster_distance)

        for it in range(self.iterations):
            for i in range(self.n_particles):    
                #recompute the particles
                new_centroids = []
                for j in range(self.k):
                    #j == the cluster in the particle
                    data = self.data
                    filter_array = np.array(cluster_values[i]) == j
                    
                    cluster_data = data[filter_array]
                    centroid = [x for x in np.mean(cluster_data, axis = 0)]
                    
                    new_centroids.append(centroid)
               
                particles[i] = new_centroids
                
                
                cluster_value = [self.get_closest_cluster(x, particles[i]) for x in self.data]
                cluster_values[i] = cluster_value
                cluster_distance = [self.get_distance_cluster(x, particles[i]) for x in self.data]
                cluster_distances[i] = cluster_distance
                
        print('fitness of particle 0:')
        print(self.get_fitness(0, cluster_values, cluster_distances))
        self.plot_particle(0, cluster_values, particles)
                
        return cluster_values, cluster_distances, particles
        


data = load_iris('data')[0] #die [0] moet bij jou nog weg
     
k_means = K_MEANS(data, iterations = 30)
cluster_values, cluster_distances, particles = k_means.run()


