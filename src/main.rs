use std::{
    collections::{HashSet, VecDeque}, // for storing  visited states
    fmt::Display,                     // pretty print
    hash::Hash,                       // To have a hashset
    rc::Rc,                           // to have parent Nodes
};

fn main() {
    let init_state = [8, 1, 3, 4, 2, 0, 7, 6, 5];
    let goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0];
    println!("init: {:?}", init_state);
    println!("running BFS:");
    run_bfs(init_state, goal_state);
    println!("running A* with h1:");
    run_a_star_with_h1(init_state, goal_state);
    println!("running A* with h2:");
    run_a_star_with_h2(init_state, goal_state);
}

fn run_bfs(init_state: [i32; 9], goal_state: [i32; 9]) {
    //get the history
    let hist = bfs(init_state, goal_state);
    //if the history is a non-null value
    if let Some(good_history) = hist {
        let history_len = good_history.len();
        println!("Total number of Nodes produced: {}", history_len);
        if history_len > 0 {
            let mut solution_path = vec![];
            let mut current_node: Option<Rc<Node>> =
                Some(good_history.get(history_len - 1).unwrap().clone());
            while let Some(node) = current_node {
                solution_path.push(node.clone());
                current_node = node.parent.clone();
            }
            println!("Total number of Nodes in solution: {}", solution_path.len());
            solution_path
                .iter()
                .rev()
                .for_each(|node| println!("{}", node.node_state));
        }
    } else {
        println!("No solution found");
    }
}

fn run_a_star_with_h1(init_state: [i32; 9], goal_state: [i32; 9]) {
    
    let hist = a_star(init_state, goal_state, h1);
    if let Some(good_history) = hist {
        let history_len = good_history.len();
        println!("Total number of Nodes produced: {}", history_len);
        if history_len > 0 {
            let mut solution_path = vec![];
            let mut current_node: Option<Rc<Node>> =
                Some(good_history.get(history_len - 1).unwrap().clone());
            while let Some(node) = current_node {
                solution_path.push(node.clone());
                current_node = node.parent.clone();
            }
            println!("Total number of Nodes in solution: {}", solution_path.len());
            solution_path
                .iter()
                .rev()
                .for_each(|node| println!("{}", node.node_state));
        }
    } else {
        println!("No solution found");
    }
}

fn run_a_star_with_h2(init_state: [i32; 9], goal_state: [i32; 9]) {
    let hist = a_star(init_state, goal_state, h2);
    if let Some(good_history) = hist {
        let history_len = good_history.len();
        println!("Total number of Nodes produced: {}", history_len);
        if history_len > 0 {
            let mut solution_path = vec![];
            let mut current_node: Option<Rc<Node>> =
                Some(good_history.get(history_len - 1).unwrap().clone());
            while let Some(node) = current_node {
                solution_path.push(node.clone());
                current_node = node.parent.clone();
            }
            println!("Total number of Nodes in solution: {}", solution_path.len());
            solution_path
                .iter()
                .rev()
                .for_each(|node| println!("{}", node.node_state));
        }
    } else {
        println!("No solution found");
    }
}
// Clone trait is for using a variable of Direction type to pass it
// as a variable without consuming it.
#[derive(Clone)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}
//pretty print
impl Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Direction::Up => write!(f, "Up"),
            Direction::Down => write!(f, "Down"),
            Direction::Left => write!(f, "Left"),
            Direction::Right => write!(f, "Right"),
        }
    }
}
struct Node {
    // An Option of reference to the parent Node,
    // Option means it can be Some(value) or None(NULL, Nil...),
    // basically a nullable value
    parent: Option<Rc<Node>>,
    //store the state in a helper struct to make the hash function simple
    node_state: NodeState,
}

#[derive(Clone)]
struct NodeState {
    // How deep we are at the path
    depth: i32,
    // Direction, also nullable because the root element's dir will be None,
    dir: Option<Direction>,
    //our actual state, as a 1 dimensional array
    state: [i32; 9],
}
//Compare two NodeState elements
impl PartialEq for NodeState {
    fn eq(&self, other: &Self) -> bool {
        self.state == other.state
    }
}
//boilerplate for Equality in NodeState
impl Eq for NodeState {}

//to Hash NodeState elements they have to be unique,
//the only field that can be unique is state
impl Hash for NodeState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.state.hash(state);
    }
}

//pretty print
impl Display for NodeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(dir) = self.dir.clone() {
            write!(
                f,
                "dir: {}, depth: {}, state: {:?}",
                dir, self.depth, self.state
            )
        } else {
            write!(
                f,
                "dir: Start, depth: {}, state: {:?}",
                self.depth, self.state
            )
        }
    }
}

//trait for processing nodes to create new paths and children
trait NodeProcessing {
    fn check_direction(&self, dir: Direction) -> Option<Rc<Node>>;
    fn get_children(&self) -> Vec<Rc<Node>>;
}

//Rc means Reference Counted, since we can't move the self value to our child,
// we have to save the reference to it
impl NodeProcessing for Rc<Node> {
    // check if we can move the empty tile up, down, left and right
    // again, using an option as the return type
    // because we might fail to move the empty tile
    // if successful, return the Some(child), else None
    fn check_direction(&self, dir: Direction) -> Option<Rc<Node>> {
        //copy the self's node and find the index of the value "0"
        let mut node_state = self.node_state.clone();
        let zero_pos = node_state.state.iter().position(|&t| t == 0).unwrap();
        // if we want to go up, we can't be in top indexes which are 0, 1 and 2
        // and if we want to go left, we cant be in indexes 0, 2 or 5 etc.
        // if contains is false for the given dir argument,
        // we can safely create a new state by swapping 2 indexes
        let contains = match dir {
            Direction::Up => [0usize, 1, 2].contains(&zero_pos),
            Direction::Down => [6usize, 7, 8].contains(&zero_pos),
            Direction::Left => [0usize, 3, 6].contains(&zero_pos),
            Direction::Right => [2usize, 5, 8].contains(&zero_pos),
        };
        //if contains is false, mutate the state copy and return new Node
        if !contains {
            match dir {
                Direction::Up => node_state.state.swap(zero_pos - 3, zero_pos),
                Direction::Down => node_state.state.swap(zero_pos, zero_pos + 3),
                Direction::Left => node_state.state.swap(zero_pos - 1, zero_pos),
                Direction::Right => node_state.state.swap(zero_pos, zero_pos + 1),
            }
            // save the direction we moved
            node_state.dir = Some(dir);
            //depth of child is 1 more than the parent's depth
            node_state.depth += 1;
            Some(Rc::new(Node {
                parent: Some(self.clone()),
                node_state,
            }))
        } else {
            //the direction is illegal, return None
            None
        }
    }

    fn get_children(&self) -> Vec<Rc<Node>> {
        // get child Nodes on each side
        let children = vec![
            self.check_direction(Direction::Down),
            self.check_direction(Direction::Left),
            self.check_direction(Direction::Up),
            self.check_direction(Direction::Right),
        ];
        //filter out the Node values and return the resulting Vec
        children.into_iter().flatten().collect()
    }
}

//BFS algorithm to go from init_state to goal_state
fn bfs(init_state: [i32; 9], goal_state: [i32; 9]) -> Option<Vec<Rc<Node>>> {
    //create root Node from the init_state
    let root = Rc::new(Node {
        parent: None, //has no parent
        node_state: NodeState {
            dir: None, //has no direction
            depth: 0,  // depth is 0 because root
            state: init_state,
        },
    });
    // save visited NodeStates in a hashset,
    // hashset keys must be unique
    let mut visited: HashSet<NodeState> = HashSet::new();
    //history of nodes that we visited
    let mut history: Vec<Rc<Node>> = Vec::new();
    //queue to pick up new nodes to process
    let mut queue = VecDeque::new();

    visited.insert(root.node_state.clone());
    queue.push_back(root);

    // while we get a value that is different than None, push it to the history
    // if its state is the goal state, return History
    // else, create children from the current node and push them back to the queue
    while let Some(current_node) = queue.pop_front() {
        history.push(current_node.clone());
        if current_node.node_state.state == goal_state {
            return Some(history);
        }
        let childs = current_node.get_children();
        for child in childs {
            if !visited.contains(&child.node_state) {
                visited.insert(child.node_state.clone());
                queue.push_back(child);
            }
        }
    }
    //No solution could be found
    None
}

//heuristic method 1:
//the number of tiles that are misplaced
//this method ignores the empty tile
fn h1(init_state: [i32; 9], goal_state: [i32; 9]) -> i32 {
    let mut h = 0;
    for i in 0..9 {
        if init_state[i] == 0 {
            continue;
        }
        if init_state[i] != goal_state[i] {
            h += 1;
        }
    }
    h
}

//heuristic method 2: manhattan distance
fn h2(init_state: [i32; 9], goal_state: [i32; 9]) -> i32 {
    let mut h = 0;
    for i in 1..9 {
        //find each tile on both states and compare their rows and columns
        let init_pos = init_state.iter().position(|&num| num == i).unwrap() as i32;
        let goal_pos = goal_state.iter().position(|&num| num == i).unwrap() as i32;
        let h_y = abs(init_pos / 3 - goal_pos / 3);
        let h_x = abs(goal_pos % 3 - init_pos % 3);
        h += h_y + h_x;
    }
    h as i32
}

//simple absolute value function
fn abs(num: i32) -> i32 {
    if num < 0 {
        return -num;
    }
    num
}

fn a_star<F>(
    init_state: [i32; 9],
    goal_state: [i32; 9],
    heurictic_method: F,
) -> Option<Vec<Rc<Node>>>
where
    F: Fn([i32; 9], [i32; 9]) -> i32,
{
    let root = Rc::new(Node {
        parent: None,
        node_state: NodeState {
            dir: None,
            depth: 0,
            state: init_state,
        },
    });

    let mut visited: HashSet<NodeState> = HashSet::new();
    let mut history: Vec<Rc<Node>> = Vec::new();
    let mut queue = VecDeque::new();

    visited.insert(root.node_state.clone());
    queue.push_back(root);

    while !queue.is_empty() {
        let mut best_choice = i32::MAX;
        for elem in queue.iter() {
            let h = heurictic_method(elem.clone().node_state.state, goal_state);
            if h == 0 {
                history.push(elem.clone());
                return Some(history);
            }
            let g = elem.node_state.depth;
            let f = g + h;
            if f < best_choice {
                best_choice = f;
            }
        }
        let mut index = usize::MAX;
        for elem in queue.iter() {
            let f =
                elem.node_state.depth + heurictic_method(elem.clone().node_state.state, goal_state);
            if f == best_choice {
                index = queue.iter().position(|x| x.node_state == elem.node_state).unwrap();
                break;
            }
        }
        history.push(queue[index].clone());

        let childs = queue[index].get_children();
        for child in childs {
            if !visited.contains(&child.node_state) {
                visited.insert(child.node_state.clone());
                queue.push_back(child);
            }
        }
        queue.remove(index);
    }
    None
}
