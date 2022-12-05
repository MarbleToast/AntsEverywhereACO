
import timeit
import numpy as np
import threading 

class ACOFacilityLocation:
    def __init__(self,
                 distance_matrix: np.array,
                 flow_matrix: np.array,
                 num_ants: int,
                 num_iterations: int,
                 evaporation_rate: float
        ):
        """
        We define:
        - distance_matrix[a1][a2] as the distance from location a1 to location a2,
        - flow_matrix[x1][x2] as the flow from facility x1 to facility x2,
        - and is_at(xN, aN) as a binary function, returning 1 if facility xN is
          assigned to location aN, and 0 if not.
          
        As defined in the specification, the cost for the edge facility x1 at
        location a1 to facility x2 at location a2 is the product of 
        distance_matrix[a1][a2] and flow_matrix[x1][x2].
        
        Therefore, we can be seen to be minimising the function:
            Σx1(Σx2(Σa1(Σa2(flow_matrix[x1][x2] * distance_matrix[a1][a2]
                * is_at(x1, a1) * is_at(x2, a2)))))
            where for each location aN, Σ‭‭‭x(is_at(x, aN)) = 1
            and where for each facility xN, Σ‭‭‭a(is_at(xN, a)) = 1
            
        On traversing an edge in the graph, the ant assigns a facility to a 
        location based on the cost function:
            C(x, a) = 

        Parameters
        ----------
        distance_matrix : np.array
            distance_matrix[a][b] represents the distance cost of traversing
            from location a to location b
        flow_matrix : np.array
            flow_matrix[x][y] represents the flow from facility x to facility y
        num_ants : int
            DESCRIPTION.
        num_iterations : int
            DESCRIPTION.
        evaporation_rate : float
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.distance_matrix = distance_matrix
        self.flow_matrix = flow_matrix
        
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.evaporation_rate = evaporation_rate
        
        self.pheromone_matrix = np.random.random_sample(distance_matrix.shape)
        
        self.ants = []
        self.fittest_path = None
        
    def run(self):
        for i in range(self.num_iterations):
            if i % 100 == 0:
                print(f"Thread {threading.get_ident()}, iteration {i}")
                
            if len(self.ants) > 0:
                current_best = min(self.ants, key=self.evaluate_fitness)
                if self.evaluate_fitness(current_best) < self.evaluate_fitness(self.fittest_path):
                     self.fittest_path = current_best
                 
            self.generate_solutions()
            self.update_pheromones()
        return self.fittest_path, self.evaluate_fitness(self.fittest_path)
            
    def generate_solutions(self):
        self.ants.clear()
        for ant in range(self.num_ants):
            path = []
            visited_nodes = []
            
            start_node = 0
            prev_node = start_node

            for i in range(len(self.distance_matrix)):
                pheromones = np.copy(self.pheromone_matrix[prev_node])
                pheromones[visited_nodes] = 0
                pheromones /= pheromones.sum()
                
                next_node = np.random.choice(range(len(self.distance_matrix)), 1, p=pheromones)[0]
                
                path.append(next_node)
                visited_nodes.append(next_node)
                prev_node = next_node

            self.ants.append(path)
            
    def evaluate_fitness(self, path):
        if path == None:
            return np.inf
        return np.sum(self.flow_matrix * self.distance_matrix[np.ix_(path, path)])
    
    def update_pheromones(self):
        for path in self.ants:
            for move in path:
                self.pheromone_matrix[move] += 1.0 / self.evaluate_fitness(path)
                
                
def run_colony(distance_matrix, flow_matrix, num_ants, evaporation_rate):
    print(f"Starting thread {threading.get_ident()}, {num_ants} ants, {evaporation_rate} delta")
    for i in range(5):
        
        colony = ACOFacilityLocation(distance_matrix, flow_matrix, num_ants, 10000, evaporation_rate)
        start = timeit.default_timer()
        res, fitness = colony.run()
        end = timeit.default_timer()
        
        with open("output.txt", "a") as f:
            f.write(f"{num_ants} ants, {evaporation_rate} rate, shortest path: {res}, fitness: {fitness}, took {int(end - start)} seconds\n")

if __name__ == "__main__":
    raw_data = None
    with open("Uni50a.dat", "r") as f:
        raw_data = f.readlines()
        
    num_facilities = int(raw_data[0].strip())
    
    distance_matrix = np.array(
        list(map(
            lambda x: list(map(
                lambda y: int(y),
                x.strip().split()
            )),
        raw_data[2:52]))
    )
    
    flow_matrix = np.array(
        list(map(
            lambda x: list(map(
                lambda y: int(y),
                x.strip().split()
            )),
        raw_data[53:]))
    )
    
    t1 = threading.Thread(target=run_colony, args=(distance_matrix, flow_matrix, 100, 0.9))
    t1.start()
    
    t2 = threading.Thread(target=run_colony, args=(distance_matrix, flow_matrix, 100, 0.5))
    t2.start()
    
    t3 = threading.Thread(target=run_colony, args=(distance_matrix, flow_matrix, 10, 0.9))
    t3.start()
    
    t4 = threading.Thread(target=run_colony, args=(distance_matrix, flow_matrix, 10, 0.5))
    t4.start()
