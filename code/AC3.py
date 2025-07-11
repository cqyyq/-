import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations
import os
from collections import deque

class MapColoringCSP:
    def __init__(self):
        # Define Australian states map
        self.variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
        # Initialize domains - WA is restricted to red, Q is restricted to green
        self.domains = {
            'WA': {'red'},     # WA is restricted to red
            'NT': {'red', 'green', 'blue'},
            'SA': {'red', 'green', 'blue'},
            'Q': {'green'},    # Q is restricted to green
            'NSW': {'red', 'green', 'blue'},
            'V': {'red', 'green', 'blue'},
            'T': {'red', 'green', 'blue'}
        }
        # Geographic constraints (edges between adjacent regions)
        self.constraints = [
            ('WA', 'NT'), ('WA', 'SA'),
            ('NT', 'SA'), ('NT', 'Q'),
            ('SA', 'Q'), ('SA', 'NSW'), ('SA', 'V'),
            ('Q', 'NSW'),
            ('NSW', 'V')
        ]
        self.graph = nx.Graph()
        self.graph.add_edges_from(self.constraints)
        self.pos = {
            'WA': (-1, 0), 'NT': (-0.5, 1), 'SA': (0, 0),
            'Q': (1, 1), 'NSW': (1, 0), 'V': (0.5, -1), 'T': (1.5, -1)
        }

        # Create output directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(current_dir, "AC3_images")
        os.makedirs(self.output_dir, exist_ok=True)

        # Color mapping for visualization
        self.color_map = {
            'red': '#FF0000',
            'green': '#00FF00',
            'blue': '#0000FF'
        }

    def revise(self, Xi, Xj):
        """Revise the domain of Xi with respect to Xj"""
        revised = False
        removed_colors = set()
        
        # Check each color in Xi's domain
        for x in list(self.domains[Xi]):
            # If there's no valid color in Xj's domain (all colors would conflict)
            if not any(x != y for y in self.domains[Xj]):
                self.domains[Xi].remove(x)
                removed_colors.add(x)
                revised = True
        
        return revised, removed_colors

    def ac3(self):
        """Run AC-3 algorithm focusing on constraint propagation"""
        step = 0
        
        # Save initial state
        self.save_state(step, "Initial state: WA=red only, Q=green only", None, None)
        step += 1

        # Initialize queue with all constraints (both directions)
        queue = deque()
        for (Xi, Xj) in self.constraints:
            queue.append((Xi, Xj))
            queue.append((Xj, Xi))

        # Keep track of domains to detect changes
        old_domains = None

        # Process all constraints
        while queue:
            Xi, Xj = queue.popleft()
            
            # Save state before revision
            domains_before = {var: set(domain) for var, domain in self.domains.items()}
            
            revised, removed_colors = self.revise(Xi, Xj)
            if revised:
                if len(self.domains[Xi]) == 0:
                    self.save_state(step, 
                                  f"No solution possible! {Xi}'s domain is empty after checking constraint with {Xj}",
                                  Xi, Xj)
                    return False
                
                # Add all constraints involving Xi back to queue
                for (Xk, Xl) in self.constraints:
                    if Xi in (Xk, Xl):
                        # Add both directions of the constraint
                        if (Xk, Xl) not in queue:
                            queue.append((Xk, Xl))
                        if (Xl, Xk) not in queue:
                            queue.append((Xl, Xk))
                
                # Save state showing the propagation
                self.save_state(step, 
                              f"Constraint propagation: {Xi} removes {removed_colors}\n"
                              f"due to {Xj} = {self.domains[Xj]}\n"
                              f"Rechecking all constraints involving {Xi}", 
                              Xi, Xj, domains_before)
                step += 1
            else:
                # Save state when no change needed
                self.save_state(step, 
                              f"Checked {Xi} against {Xj}: No changes needed\n"
                              f"{Xi} keeps colors {self.domains[Xi]}", 
                              Xi, Xj)
                step += 1

            # Store current domains for next comparison
            new_domains = {var: set(domain) for var, domain in self.domains.items()}
            
            # If domains haven't changed after a full cycle through the queue
            if old_domains == new_domains and len(queue) == 0:
                break
            old_domains = new_domains

        return True

    def save_state(self, step, message, current_var=None, next_var=None, domains_before=None):
        """Save the current state as an image with detailed information"""
        plt.figure(figsize=(12, 8))
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, self.pos, edge_color='gray')
        
        # Draw nodes and their domains
        for var in self.variables:  # Use self.variables to maintain order
            x, y = self.pos[var]
            
            # Draw node circle with appropriate highlighting
            if var == current_var:
                circle = plt.Circle((x, y), 0.15, color='yellow', alpha=0.3)
            elif var == next_var:
                circle = plt.Circle((x, y), 0.15, color='lightgreen', alpha=0.3)
            elif var == 'WA':
                circle = plt.Circle((x, y), 0.15, color='lightpink', alpha=0.3)
            elif var == 'Q':
                circle = plt.Circle((x, y), 0.15, color='lightgreen', alpha=0.3)
            else:
                circle = plt.Circle((x, y), 0.15, color='lightgray', alpha=0.3)
            plt.gca().add_patch(circle)
            
            # Draw node label
            plt.text(x, y, var, horizontalalignment='center', verticalalignment='center',
                    fontsize=12, fontweight='bold')
            
            # Show current domain
            domain_text = f"{var}'s colors:\n{', '.join(sorted(self.domains[var]))}"
            plt.text(x, y-0.2, domain_text, horizontalalignment='center',
                    verticalalignment='top', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7))
            
            # Show removed colors if any
            if domains_before and var in domains_before:
                removed = domains_before[var] - self.domains[var]
                if removed:
                    removed_text = f"Removed: {', '.join(sorted(removed))}"
                    plt.text(x, y-0.4, removed_text, horizontalalignment='center',
                            verticalalignment='top', color='red', fontsize=8)

            # Draw colored squares for available colors
            color_x = x - 0.1 * (len(self.domains[var]) - 1)
            for color in sorted(self.domains[var]):
                plt.scatter(color_x, y-0.1, c=self.color_map[color], s=100)
                color_x += 0.2

        # Highlight current constraint being checked
        if current_var and next_var:
            if (current_var, next_var) in self.constraints or (next_var, current_var) in self.constraints:
                nx.draw_networkx_edges(self.graph, self.pos,
                                     edgelist=[(current_var, next_var)],
                                     edge_color='red', width=2)

        # Add step information
        plt.title(f"Step {step}:\n{message}", pad=20, wrap=True)
        plt.axis('off')
        plt.axis('equal')
        
        # Save image
        plt.savefig(os.path.join(self.output_dir, f'step_{step}.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()

    def solve(self):
        """Run AC-3 algorithm and display results"""
        print("\nRunning AC-3 constraint propagation...")
        print("\nInitial domains:")
        for var in self.variables:  # Use self.variables to maintain order
            print(f"{var}: {sorted(self.domains[var])}")
            
        if self.ac3():
            print("\nConstraint propagation completed successfully!")
            print("\nFinal domains after propagation:")
            for var in self.variables:  # Use self.variables to maintain order
                print(f"{var}: {sorted(self.domains[var])}")
        else:
            print("\nConstraint propagation detected inconsistency!")
            
        print(f"\nVisualization steps saved to: {os.path.abspath(self.output_dir)}")

# Run the solver
solver = MapColoringCSP()
solver.solve()