use crate::common::{
    ArchivedAdj, ArchivedEdge, ArchivedNode, ArchivedOffsets, ArchivedTableType, Offsets,
};
use clap::Parser;
use half::bf16;
use itertools::izip;
use memmap2::Mmap;
use numpy::PyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::PyObject;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::{pyclass, pymethods};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::seq::index;
use rkyv::rancor::Error;
use rkyv::vec::ArchivedVec;
use std::collections::HashMap;
use std::env::var;
use std::fs;
use std::io::{BufReader, Read};
use std::str;
use std::time::Instant;

const MAX_F2P_NBRS: usize = 5;

struct Vecs {
    node_idxs: Vec<i32>,
    f2p_nbr_idxs: Vec<i32>,
    table_name_idxs: Vec<i32>,
    col_name_idxs: Vec<i32>,
    class_value_idxs: Vec<i32>,
    col_name_values: Vec<bf16>,
    sem_types: Vec<i32>,
    number_values: Vec<bf16>,
    text_values: Vec<bf16>,
    datetime_values: Vec<bf16>,
    boolean_values: Vec<bf16>,
    masks: Vec<bool>,
    is_targets: Vec<bool>,
    is_task_nodes: Vec<bool>,
    is_padding: Vec<bool>,
    timestamps: Vec<i32>,
    true_batch_size: usize,
}

struct Slices<'a> {
    node_idxs: &'a mut [i32],
    f2p_nbr_idxs: &'a mut [i32],
    table_name_idxs: &'a mut [i32],
    col_name_idxs: &'a mut [i32],
    class_value_idxs: &'a mut [i32],
    col_name_values: &'a mut [bf16],
    sem_types: &'a mut [i32],
    number_values: &'a mut [bf16],
    text_values: &'a mut [bf16],
    datetime_values: &'a mut [bf16],
    boolean_values: &'a mut [bf16],
    masks: &'a mut [bool],
    is_targets: &'a mut [bool],
    is_task_nodes: &'a mut [bool],
    is_padding: &'a mut [bool],
    timestamps: &'a mut [i32],
}

impl Vecs {
    fn new(batch_size: usize, seq_len: usize, true_batch_size: usize, d_text: usize) -> Self {
        let l = batch_size * seq_len;
        Self {
            node_idxs: vec![-1; l],
            f2p_nbr_idxs: vec![-1; l * MAX_F2P_NBRS],
            table_name_idxs: vec![0; l],
            col_name_idxs: vec![0; l],
            class_value_idxs: vec![-1; l],
            col_name_values: vec![bf16::ZERO; l * d_text],
            sem_types: vec![0; l],
            number_values: vec![bf16::ZERO; l],
            text_values: vec![bf16::ZERO; l * d_text],
            datetime_values: vec![bf16::ZERO; l],
            boolean_values: vec![bf16::ZERO; l],
            masks: vec![false; l],
            is_targets: vec![false; l],
            is_task_nodes: vec![false; l],
            is_padding: vec![true; l],
            timestamps: vec![i32::MIN; l],
            true_batch_size,
        }
    }

    fn chunks_exact_mut(
        &mut self,
        seq_len: usize,
        d_text: usize,
    ) -> impl Iterator<Item = Slices<'_>> {
        izip!(
            self.node_idxs.chunks_exact_mut(seq_len),
            self.f2p_nbr_idxs.chunks_exact_mut(seq_len * MAX_F2P_NBRS),
            self.table_name_idxs.chunks_exact_mut(seq_len),
            self.col_name_idxs.chunks_exact_mut(seq_len),
            self.class_value_idxs.chunks_exact_mut(seq_len),
            self.col_name_values.chunks_exact_mut(seq_len * d_text),
            self.sem_types.chunks_exact_mut(seq_len),
            self.number_values.chunks_exact_mut(seq_len),
            self.text_values.chunks_exact_mut(seq_len * d_text),
            self.datetime_values.chunks_exact_mut(seq_len),
            self.boolean_values.chunks_exact_mut(seq_len),
            self.masks.chunks_exact_mut(seq_len),
            self.is_targets.chunks_exact_mut(seq_len),
            self.is_task_nodes.chunks_exact_mut(seq_len),
            self.is_padding.chunks_exact_mut(seq_len),
            self.timestamps.chunks_exact_mut(seq_len)
        )
        .map(
            |(
                node_idxs,
                f2p_nbr_idxs,
                table_name_idxs,
                col_name_idxs,
                class_value_idxs,
                col_name_values,
                sem_types,
                number_values,
                text_values,
                datetime_values,
                boolean_values,
                masks,
                is_targets,
                is_task_nodes,
                is_padding,
                timestamps,
            )| Slices {
                node_idxs,
                f2p_nbr_idxs,
                table_name_idxs,
                col_name_idxs,
                class_value_idxs,
                col_name_values,
                sem_types,
                number_values,
                text_values,
                datetime_values,
                boolean_values,
                masks,
                is_targets,
                is_task_nodes,
                is_padding,
                timestamps,
            },
        )
    }
    fn into_pyobject<'a>(self, py: Python<'a>) -> PyResult<Vec<PyObject>> {
        Ok(vec![
            ("node_idxs", PyArray1::from_vec(py, self.node_idxs))
                .into_py_any(py)
                .unwrap(),
            ("f2p_nbr_idxs", PyArray1::from_vec(py, self.f2p_nbr_idxs))
                .into_py_any(py)
                .unwrap(),
            (
                "table_name_idxs",
                PyArray1::from_vec(py, self.table_name_idxs),
            )
                .into_py_any(py)
                .unwrap(),
            ("col_name_idxs", PyArray1::from_vec(py, self.col_name_idxs))
                .into_py_any(py)
                .unwrap(),
            (
                "class_value_idxs",
                PyArray1::from_vec(py, self.class_value_idxs),
            )
                .into_py_any(py)
                .unwrap(),
            (
                "col_name_values",
                PyArray1::from_vec(py, self.col_name_values),
            )
                .into_py_any(py)
                .unwrap(),
            ("sem_types", PyArray1::from_vec(py, self.sem_types))
                .into_py_any(py)
                .unwrap(),
            ("number_values", PyArray1::from_vec(py, self.number_values))
                .into_py_any(py)
                .unwrap(),
            ("text_values", PyArray1::from_vec(py, self.text_values))
                .into_py_any(py)
                .unwrap(),
            (
                "datetime_values",
                PyArray1::from_vec(py, self.datetime_values),
            )
                .into_py_any(py)
                .unwrap(),
            (
                "boolean_values",
                PyArray1::from_vec(py, self.boolean_values),
            )
                .into_py_any(py)
                .unwrap(),
            ("masks", PyArray1::from_vec(py, self.masks))
                .into_py_any(py)
                .unwrap(),
            ("is_targets", PyArray1::from_vec(py, self.is_targets))
                .into_py_any(py)
                .unwrap(),
            ("is_task_nodes", PyArray1::from_vec(py, self.is_task_nodes))
                .into_py_any(py)
                .unwrap(),
            ("is_padding", PyArray1::from_vec(py, self.is_padding))
                .into_py_any(py)
                .unwrap(),
            ("timestamps", PyArray1::from_vec(py, self.timestamps))
                .into_py_any(py)
                .unwrap(),
            ("true_batch_size", self.true_batch_size)
                .into_py_any(py)
                .unwrap(),
        ])
    }
}

struct Dataset {
    mmap: Mmap,
    text_mmap: Mmap,
    p2f_adj_mmap: Mmap,
    offsets: Vec<i64>,
}

struct Item {
    dataset_idx: i32,
    node_idx: i32,
}

#[pyclass]
pub struct Sampler {
    batch_size: usize,
    rank: usize,
    world_size: usize,
    datasets: Vec<Dataset>,
    items: Vec<Item>,
    ctx_len: usize,
    max_bfs_width: usize,
    epoch: u64,
    d_text: usize,
    seed: u64,
    target_columns: Vec<i32>,
    columns_to_drop: Vec<Vec<i32>>,
    max_items_per_task: i64,
    dataset_tuples: Vec<(String, i32, i32)>, // (db_name, node_idx_offset, num_nodes) for each dataset
}

#[pymethods]
impl Sampler {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dataset_tuples: Vec<(String, i32, i32)>,
        batch_size: usize,
        rank: usize,
        world_size: usize,
        ctx_len: usize,
        max_bfs_width: usize,
        embedding_model: &str,
        d_text: usize,
        seed: u64,
        target_columns: Vec<i32>,
        columns_to_drop: Vec<Vec<i32>>,
        max_items_per_task: i64,
    ) -> Self {
        let mut datasets = Vec::new();

        for (db_name, _node_idx_offset, _num_nodes) in dataset_tuples.iter() {
            let pre_path = format!("{}/scratch/pre/{}", var("HOME").unwrap(), db_name);
            let nodes_path = format!("{}/nodes.rkyv", pre_path);
            let file = fs::File::open(&nodes_path).unwrap();
            let mmap = unsafe { Mmap::map(&file).unwrap() };

            let text_path = format!("{}/text_emb_{}.bin", pre_path, embedding_model);
            let text_file = fs::File::open(&text_path).unwrap();
            let text_mmap = unsafe { Mmap::map(&text_file).unwrap() };

            let offsets_path = format!("{}/offsets.rkyv", pre_path);
            let file = fs::File::open(&offsets_path).unwrap();
            let mut bytes = Vec::new();
            BufReader::new(file).read_to_end(&mut bytes).unwrap();
            let archived = rkyv::access::<ArchivedOffsets, Error>(&bytes).unwrap();
            let offsets = rkyv::deserialize::<Offsets, Error>(archived).unwrap();
            let offsets = offsets.offsets;

            let p2f_adj_path = format!("{}/p2f_adj.rkyv", pre_path);
            let p2f_adj_file = fs::File::open(&p2f_adj_path).unwrap();
            let p2f_adj_mmap = unsafe { Mmap::map(&p2f_adj_file).unwrap() };

            datasets.push(Dataset {
                mmap,
                text_mmap,
                p2f_adj_mmap,
                offsets,
            });
        }

        let epoch = 0;
        let mut sampler = Self {
            batch_size,
            rank,
            world_size,
            datasets,
            items: Vec::new(), // Will be populated by create_items
            ctx_len,
            max_bfs_width,
            epoch,
            d_text,
            seed,
            target_columns,
            columns_to_drop,
            max_items_per_task,
            dataset_tuples,
        };

        sampler.create_items(0);
        sampler
    }

    fn len_py(&self) -> PyResult<usize> {
        Ok(self.len())
    }

    fn batch_py<'a>(&self, py: Python<'a>, batch_idx: usize) -> PyResult<Vec<PyObject>> {
        self.batch(batch_idx).into_pyobject(py)
    }

    fn shuffle_py(&mut self, epoch: u64) {
        self.epoch = epoch;
        self.create_items(epoch);
    }
}

impl Sampler {
    /// Rebuild items list with epoch-based random sampling (so also shuffling the items).
    fn create_items(&mut self, epoch: u64) {
        self.items.clear();

        let num_tasks = self.dataset_tuples.len() as u64;

        for (i, &(_, node_idx_offset, num_nodes)) in self.dataset_tuples.iter().enumerate() {
            let target = self.target_columns[i];

            let num_to_sample = if self.max_items_per_task == -1 {
                num_nodes as usize
            } else {
                (num_nodes as usize).min(self.max_items_per_task as usize)
            };

            let rng_seed = self.seed + (epoch * num_tasks) + (i as u64);
            let mut rng = StdRng::seed_from_u64(rng_seed);

            let sampled_indices = index::sample(&mut rng, num_nodes as usize, num_to_sample);

            for idx in sampled_indices.iter() {
                let node_idx = node_idx_offset + idx as i32;
                let node = get_node(&self.datasets[i], node_idx);

                // Skip if node doesn't have target column
                if node.col_name_idxs.iter().any(|&c| c == target) {
                    self.items.push(Item {
                        dataset_idx: i as i32,
                        node_idx,
                    });
                }
            }
        }

        // Shuffle the entire items list
        let mut rng = StdRng::seed_from_u64((self.seed << 32) | (epoch & 0xFFFFFFFF));
        self.items.shuffle(&mut rng);
    }

    fn len(&self) -> usize {
        self.items.len().div_ceil(self.batch_size * self.world_size)
    }

    fn batch(&self, batch_idx: usize) -> Vecs {
        let true_batch_size = self.batch_size.min(
            self.items.len()
                - self.rank * self.batch_size
                - batch_idx * self.batch_size * self.world_size,
        );

        let mut vecs = Vecs::new(self.batch_size, self.ctx_len, true_batch_size, self.d_text);

        // Parallelize batch processing across sequences
        vecs.chunks_exact_mut(self.ctx_len, self.d_text)
            .enumerate()
            .for_each(|(i, slices)| {
                let j =
                    batch_idx * self.batch_size * self.world_size + self.rank * self.batch_size + i;
                // when self.batch_size > true_batch_size, this will wrap around
                let j = j % self.items.len();
                let item = &self.items[j];
                self.seq(item, slices);
            });
        vecs
    }

    fn seq(&self, item: &Item, mut slices: Slices) {
        let dataset = &self.datasets[item.dataset_idx as usize];
        let target_column = self.target_columns[item.dataset_idx as usize];
        let columns_to_drop = &self.columns_to_drop[item.dataset_idx as usize];

        let target_node_idx = item.node_idx;
        let target_node = get_node(dataset, target_node_idx);

        let mut visited = std::collections::HashSet::new();
        let mut visited_at_depth: HashMap<i32, usize> = HashMap::new();

        let mut cells_to_add: Vec<(i32, usize, i32)> = Vec::new();

        let mut rng = StdRng::seed_from_u64(
            ((self.seed << 32) | (self.epoch & 0xFFFFFFFF)) ^ (target_node_idx as u64),
        );
        let bfs_nodes = self.bfs_collect_nodes(
            dataset,
            target_node_idx,
            &mut rng,
            self.ctx_len,
            &mut visited_at_depth,
        );

        for bfs_node_idx in bfs_nodes {
            if visited.contains(&bfs_node_idx) {
                continue;
            }
            visited.insert(bfs_node_idx);

            let node = get_node(dataset, bfs_node_idx);
            for cell_i in 0..node.col_name_idxs.len() {
                let col_idx: i32 = node.col_name_idxs[cell_i].into();

                // Skip columns to drop
                if (node.node_idx == target_node_idx && columns_to_drop.contains(&col_idx))
                    || (node.timestamp == target_node.timestamp
                        && columns_to_drop.contains(&col_idx))
                {
                    continue;
                }

                cells_to_add.push((bfs_node_idx, cell_i, col_idx));

                if cells_to_add.len() == self.ctx_len {
                    break;
                }
            }

            if cells_to_add.len() == self.ctx_len {
                break;
            }
        }

        // Sort by column index
        cells_to_add.sort_by_key(|&(_, _, col_idx)| col_idx);

        // Add cells to sequence in sorted order
        let mut seq_i = 0;
        for (node_idx, cell_i, _col_idx) in cells_to_add.iter() {
            if seq_i >= self.ctx_len {
                break;
            }

            self.add_single_cell(
                dataset,
                *node_idx,
                *cell_i,
                target_node_idx,
                target_column,
                &mut seq_i,
                &mut slices,
            );
        }
    }

    /// Add a single cell from a node to the sequence.
    #[allow(clippy::too_many_arguments)]
    fn add_single_cell(
        &self,
        dataset: &Dataset,
        node_idx: i32,
        cell_i: usize,
        target_node_idx: i32,
        target_column: i32,
        seq_i: &mut usize,
        slices: &mut Slices,
    ) {
        let node = get_node(dataset, node_idx);

        slices.node_idxs[*seq_i] = node.node_idx.into();

        assert!(node.f2p_nbr_idxs.len() <= MAX_F2P_NBRS);
        for (j, f2p_nbr_idx) in node.f2p_nbr_idxs.iter().enumerate() {
            slices.f2p_nbr_idxs[*seq_i * MAX_F2P_NBRS + j] = f2p_nbr_idx.into();
        }

        slices.table_name_idxs[*seq_i] = node.table_name_idx.into();
        slices.col_name_idxs[*seq_i] = node.col_name_idxs[cell_i].into();
        slices.class_value_idxs[*seq_i] = node.class_value_idx[cell_i].into();
        slices.col_name_values[*seq_i * self.d_text..(*seq_i + 1) * self.d_text].copy_from_slice(
            get_text_emb(dataset, slices.col_name_idxs[*seq_i], self.d_text),
        );

        slices.sem_types[*seq_i] = node.sem_types[cell_i].clone() as i32;
        slices.number_values[*seq_i] = bf16::from_f32(node.number_values[cell_i].into());

        let text_idx: i32 = node.text_values[cell_i].into();
        slices.text_values[*seq_i * self.d_text..(*seq_i + 1) * self.d_text]
            .copy_from_slice(get_text_emb(dataset, text_idx, self.d_text));

        slices.datetime_values[*seq_i] = bf16::from_f32(node.datetime_values[cell_i].into());
        slices.boolean_values[*seq_i] = bf16::from_f32(node.boolean_values[cell_i].into());

        slices.is_targets[*seq_i] =
            node.node_idx == target_node_idx && node.col_name_idxs[cell_i] == target_column;
        slices.masks[*seq_i] = slices.is_targets[*seq_i];

        slices.is_task_nodes[*seq_i] =
            node.is_task_node || (node.col_name_idxs[cell_i] == target_column);
        slices.is_padding[*seq_i] = false;
        slices.timestamps[*seq_i] = match node.timestamp.as_ref() {
            Some(ts) => (*ts).into(),
            None => i32::MIN,
        };

        *seq_i += 1;
    }

    /// Performs BFS to collect nodes for context.
    fn bfs_collect_nodes(
        &self,
        dataset: &Dataset,
        start_idx: i32,
        rng: &mut StdRng,
        max_cells: usize,
        visited_at_depth: &mut HashMap<i32, usize>,
    ) -> Vec<i32> {
        let mut result = Vec::new();

        let start_node = get_node(dataset, start_idx);
        let mut num_cells = 0;

        // Two frontier data structures:
        // f2p_ftr: stack of (depth, node_idx) for f2p edges
        // p2f_ftr: vector of vectors, one per depth level, for p2f edges
        let mut f2p_ftr: Vec<(usize, i32)> = Vec::new();
        let mut p2f_ftr: Vec<Vec<i32>> = vec![vec![start_idx]];

        loop {
            // Select node
            let (depth, node_idx) = if !f2p_ftr.is_empty() {
                f2p_ftr.pop().unwrap()
            } else {
                let mut depth_choices = Vec::new();
                for (i, node) in p2f_ftr.iter().enumerate() {
                    if !node.is_empty() {
                        depth_choices.push(i);
                    }
                }
                if depth_choices.is_empty() {
                    return result;
                } else {
                    let depth = depth_choices[0];
                    let r = rng.random_range(0..p2f_ftr[depth].len());
                    let l = p2f_ftr[depth].len();
                    p2f_ftr[depth].swap(r, l - 1);
                    let node_idx = p2f_ftr[depth].pop().unwrap();
                    (depth, node_idx)
                }
            };

            // Check if node was visited at a depth <= current depth
            if let Some(&prev_depth) = visited_at_depth.get(&node_idx)
                && prev_depth <= depth
            {
                continue;
            }

            let node = get_node(dataset, node_idx);

            // Update number of cells collected
            num_cells += node.col_name_idxs.len();
            if num_cells >= max_cells {
                return result;
            }

            // Record the depth at which this node was visited
            visited_at_depth.insert(node_idx, depth);

            result.push(node_idx);

            // Add f2p edges to f2p frontier
            for edge in node.f2p_edges.iter() {
                f2p_ftr.push((depth + 1, edge.node_idx.into()));
            }

            // Get p2f edges and process them
            let p2f_edges = get_p2f_edges(dataset, node_idx);

            // Temporary storage for db edges to be subsampled
            let mut db_p2f_ftr: Vec<i32> = Vec::new();

            // The edges are sorted by timestamp, so we can binary search to find valid ones
            let valid_edges = p2f_edges.as_slice().partition_point(|edge| {
                edge.timestamp.is_none()
                    || (start_node.timestamp.is_some() && edge.timestamp <= start_node.timestamp)
            });

            // Filter valid edges by table constraints
            let p2f_edges = &p2f_edges.as_slice()[..valid_edges];

            for edge in p2f_edges.iter() {
                // include edges to task table only if seed node belongs to the task table
                if edge.table_name_idx != start_node.table_name_idx
                    && edge.table_type != ArchivedTableType::Db
                {
                    continue;
                }

                if edge.table_type == ArchivedTableType::Db {
                    db_p2f_ftr.push(edge.node_idx.into());
                    continue;
                }

                if depth + 1 >= p2f_ftr.len() {
                    for _i in p2f_ftr.len()..=depth + 1 {
                        p2f_ftr.push(vec![]);
                    }
                }
                p2f_ftr[depth + 1].push(edge.node_idx.into());
            }

            // Subsample DB edges based on max_bfs_width
            let idxs = if db_p2f_ftr.len() > self.max_bfs_width {
                index::sample(rng, db_p2f_ftr.len(), self.max_bfs_width).into_vec()
            } else {
                (0..db_p2f_ftr.len()).collect::<Vec<_>>()
            };

            for idx in idxs.iter() {
                if depth + 1 >= p2f_ftr.len() {
                    for _i in p2f_ftr.len()..=depth + 1 {
                        p2f_ftr.push(vec![]);
                    }
                }
                p2f_ftr[depth + 1].push(db_p2f_ftr[*idx]);
            }
        }
    }
}

fn get_node(dataset: &Dataset, idx: i32) -> &ArchivedNode {
    let l = dataset.offsets[idx as usize] as usize;
    let r = dataset.offsets[(idx + 1) as usize] as usize;
    let bytes = &dataset.mmap[l..r];
    unsafe { rkyv::access_unchecked::<ArchivedNode>(bytes) }
}

fn get_p2f_edges(dataset: &Dataset, idx: i32) -> &ArchivedVec<ArchivedEdge> {
    let bytes = &dataset.p2f_adj_mmap[..];
    let p2f_adj = unsafe { rkyv::access_unchecked::<ArchivedAdj>(bytes) };
    &p2f_adj.adj[idx as usize]
}

fn get_text_emb(dataset: &Dataset, idx: i32, d_text: usize) -> &[bf16] {
    let (pref, text_emb, suf) = unsafe { dataset.text_mmap.align_to::<bf16>() };
    assert!(pref.is_empty() && suf.is_empty());
    &text_emb[(idx as usize) * d_text..(idx as usize + 1) * d_text]
}

#[derive(Parser)]
pub struct Cli {
    #[arg(default_value = "rel-f1")]
    db_name: String,
    #[arg(default_value = "128")]
    batch_size: usize,
    #[arg(default_value = "1024")]
    seq_len: usize,
    #[arg(default_value = "1000")]
    num_trials: usize,
}

pub fn main(cli: Cli) {
    let tic = Instant::now();
    let sampler = Sampler::new(
        vec![(cli.db_name, 0, 10)], // dataset_tuples
        cli.batch_size,             // batch_size
        0,                          // rank
        1,                          // world_size
        cli.seq_len,                // ctx_len
        128,                        // max_bfs_width
        "all-MiniLM-L12-v2",        // embedding_model
        384,                        // d_text
        0,                          // seed
        vec![-1; 1],                // target_columns
        vec![Vec::<i32>::new()],    // columns_to_drop
        -1,                         // max_items_per_task (no limit)
    );
    println!("Sampler loaded in {:?}", tic.elapsed());

    let mut sum = 0;
    let mut sum_sq = 0;
    let mut rng = rand::rng();
    for _ in 0..cli.num_trials {
        let tic = Instant::now();
        let batch_idx = rng.random_range(0..sampler.len());
        let _batch = sampler.batch(batch_idx);
        let elapsed = tic.elapsed().as_millis();
        sum += elapsed;
        sum_sq += elapsed * elapsed;
    }
    let mean = sum as f64 / cli.num_trials as f64;
    let std = (sum_sq as f64 / cli.num_trials as f64 - mean * mean).sqrt();
    println!("Mean: {} ms,\tStd: {} ms", mean, std);
}
