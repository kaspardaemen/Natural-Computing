import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
np.random.seed(13)
from sklearn.datasets import load_iris

class K_MEANS ():
    def __init__ (self, data, w=0.73, v=[10], a=1.5, b=1.5, pso = False, k =3, iterations = 1, n_particles = 1):
        self.data = data
        self. pso = pso
        #number of centroids per particle
        self.k = k
        self.iterations = iterations
        #number of particles
        self.n_particles = n_particles
        
        self.w = w
        self.v = v
        self.a = a
        self.b = b
        
    def initialize_particles (self): 
        n = self.n_particles 
        particles = []
        cluster_values = []
        cluster_distances = []
        for i in range(n):
            #sample a random value
            np.random.seed(10+i)
            index = np.random.choice(len(self.data), self.k, replace=False) 
            print(f'index: {index}')
            particles.append(self.data[index])
            
            cluster_value = [self.get_closest_cluster(x, particles[i]) for x in self.data]
            cluster_values.append(cluster_value)
            cluster_distance = [self.get_distance_cluster(x, particles[i]) for x in self.data]
            cluster_distances.append(cluster_distance)
        
        return list(zip(particles, cluster_values, cluster_distances))
    
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
    
    def get_fitness(self, particle):
        centroids, cluster_values, distances = particle
       
        means = []
        for i in range(self.k):
            #j == the cluster in the particle
            data = self.data
            filter_array = np.array(cluster_values) == i
            distances = data[filter_array]
            
            average_distance = np.mean(distances)
            
            means.append(average_distance * len(distances)/len(data))
        return sum(means)
    
    def plot_particle(self, particle):
        centroids, cluster_values, distances = particle
        
        centroids = pd.DataFrame(centroids)
        df = pd.DataFrame(data=self.data)
        df['closest'] = cluster_values
       
        sb.scatterplot(data = df, x = 0, y = 1, hue='closest')
        sb.scatterplot(data =  centroids, x = 0, y = 1, color= 'red')
        plt.show()
        
        plt.show()
        
    def update(self):
        # r1 = np.random.random()
        # r2 = np.random.random()
        r1 = 0.5
        r2 = 0.5
        
        
        for i in range(len(self.particles)):
            centroids, values, distances = self.particles[i]
            best_centroids, _, _ = self.best_particles[i]
            
            term1 = self.w * self.v[i] 
            term2 =  self.a * r1  * (np.array(best_centroids) - np.array(centroids))
            term3 = self.a * r2  * (np.array(best_centroids) - np.array(centroids))
            self.v[i] = term1 + term2 + term3
            
            # print(f'term1: {term1}')
            # print(f'term2: {term2}')
            # print(f'term3: {term3}')
            
            new_centroids = centroids + self.v[i]
            
            
            cluster_value = [self.get_closest_cluster(x, new_centroids) for x in self.data]
            cluster_distance = [self.get_distance_cluster(x, new_centroids) for x in self.data]
            self.particles[i] = (new_centroids, cluster_value, cluster_distance)
    
    def run(self):
        #Each particle is a tuple: (centroids, values, distances)
        self.particles = self.initialize_particles()
        self.best_particles = self.particles.copy()
        self.best_global = self.particles[0]

  
        # self.cluster_values = []
        # self.cluster_distances = []
    

        for it in range(self.iterations):
            for i in range(len(self.particles)):
                centroids, cluster_values, cluster_distances = self.particles[i]
                #recompute the particles
                new_centroids = []
                for j in range(self.k):
                    #j == the cluster in the particle
                    data = self.data
                    filter_array = np.array(cluster_values) == j
                    
                    cluster_data = data[filter_array]
                    centroid = [x for x in np.mean(cluster_data, axis = 0)]
                    
                    new_centroids.append(centroid)
               
                cluster_value = [self.get_closest_cluster(x, new_centroids) for x in self.data]
                cluster_distance = [self.get_distance_cluster(x, new_centroids) for x in self.data]
                self.particles[i] = (new_centroids, cluster_value, cluster_distance)
            
                #UPDATE LOCAL BEST
                condition = self.get_fitness(self.particles[i]) < self.get_fitness(self.best_particles[i]) 
                self.best_particles[i] = self.particles[i] if condition else  self.best_particles[i] 
            best_index = np.argmin([self.get_fitness(x) for x in self.best_particles])
            self.best_global = self.best_particles[best_index]
            self.update()
            #print(self.particles[0])
        self.plot_particle(self.best_global)
        self.plot_particle(self.particles[0])
        print(f'fitness: {self.get_fitness(self.best_global)}')
        
                
        # return self.cluster_values, self.cluster_distances, self.particles
        return self.best_global
        


data = load_iris('data')[0] #die [0] moet bij jou nog weg
     
k_means = K_MEANS(data, iterations = 9, n_particles = 3, v = [0.5,0.6,0.4])
particle = k_means.run()


