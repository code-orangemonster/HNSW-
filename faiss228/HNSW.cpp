/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "HNSW.h"
#include<iostream>
#include<fstream>
#include<algorithm>
#include<math.h>
#include<unordered_map>
#include<map>
#include<set>
#include<random>
#include<chrono>

namespace faiss {

using idx_t = Index::idx_t;
using DistanceComputer = HNSW::DistanceComputer;

/**************************************************************
 * HNSW structure implementation
 **************************************************************/

// 获取当前层中的邻居数目
int HNSW::nb_neighbors(int layer_no) const
{
  return cum_nneighbor_per_level[layer_no + 1] -
    cum_nneighbor_per_level[layer_no];
}

// 
void HNSW::set_nb_neighbors(int level_no, int n)
{
  FAISS_THROW_IF_NOT(levels.size() == 0);
  int cur_n = nb_neighbors(level_no);
  for (int i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
    cum_nneighbor_per_level[i] += n - cur_n;
  }
}

int HNSW::cum_nb_neighbors(int layer_no) const
{
  return cum_nneighbor_per_level[layer_no];
}

// void HNSW::neigh(Index* ind, idx_t no){
//   for (int i = 0; i < count; ++i)
//   {
//     neighbor_range(0,0,begin,end);

//   }
// }

void HNSW::neighbor_range(idx_t no, int layer_no,
                          size_t * begin, size_t * end) const
{
  // offsets中存放的是当前节点no 元素所有邻居在neighbors中的位置
  // cum_nb_neighbors中存放的是截止到当前层邻居的总和 
  // cum_nb_neighbors(layer_no)表示layer层以上邻居总和
  // cum_nb_neighbors(layer_no+1)表示layer-1层以上邻居总和
  // end - begin就是当前节点（通过offsets找到）在当前层邻居（通过cum_nb_neighbors）的位置
  size_t o = offsets[no];
  *begin = o + cum_nb_neighbors(layer_no);
  *end = o + cum_nb_neighbors(layer_no + 1);
}



// HNSW::HNSW(int M) : rng(std::random_device {} ()) {
HNSW::HNSW(int M) : rng(12345) {
  set_default_probas(M, 1.0 / log(M));
  max_level = -1;
  entry_point = -1;
  efSearch = 16;
  efConstruction = 40;
  upper_beam = 1;
  offsets.push_back(0);
  hot_hubs.resize(1);
  globleHeatDegrees.clear();
}


int HNSW::random_level()
{
  // assign_probas
  // 0.875000 \ 0.109375 \ 0.013672 \ 0.001709 \ 0.000214 \ 0.000027 \ 0.000003 \ 0.000000

  double f = rng.rand_float();
  // printf("%f\n",f)
  // could be a bit faster with bissection
  for (int level = 0; level < assign_probas.size(); level++) {
    if (f < assign_probas[level]) {
      return level;
    }
    f -= assign_probas[level];
  }
  // happens with exponentially low probability
  return assign_probas.size() - 1;
}

// set_default_probas(M, 1.0 / log(M));
void HNSW::set_default_probas(int M, float levelMult)
{
  int nn = 0;
  cum_nneighbor_per_level.push_back (0);
  for (int level = 0; ;level++) {
    float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
    if (proba < 1e-9) break;
    assign_probas.push_back(proba);
    nn += level == 0 ? M * 2 : M;
    // cum_nneighbor_per_level中内容： 0 2M 3M 4M 5M
    cum_nneighbor_per_level.push_back (nn);
  }
}

// neighbor_tables 就是存放存放图结构的表，实际上是一个一维表neighbors
void HNSW::clear_neighbor_tables(int level)
{
  for (int i = 0; i < levels.size(); i++) {
    size_t begin, end;
    neighbor_range(i, level, &begin, &end);
    for (size_t j = begin; j < end; j++) {
      neighbors[j] = -1;
    }
  }
}


void HNSW::reset() {
  max_level = -1;
  entry_point = -1;
  offsets.clear();
  offsets.push_back(0);
  levels.clear();
  neighbors.clear();
}



void HNSW::print_neighbor_stats(int level) const
{
  FAISS_THROW_IF_NOT (level < cum_nneighbor_per_level.size());
  printf("stats on level %d, max %d neighbors per vertex:\n",
         level, nb_neighbors(level));
  size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
  reduction(+: tot_reciprocal) reduction(+: n_node)
  for (int i = 0; i < levels.size(); i++) {
    if (levels[i] > level) {
      n_node++;
      size_t begin, end;
      neighbor_range(i, level, &begin, &end);
      std::unordered_set<int> neighset;
      for (size_t j = begin; j < end; j++) {
        if (neighbors [j] < 0) break;
        neighset.insert(neighbors[j]);
      }
      int n_neigh = neighset.size();
      int n_common = 0;
      int n_reciprocal = 0;
      for (size_t j = begin; j < end; j++) {
        storage_idx_t i2 = neighbors[j];
        if (i2 < 0) break;
        FAISS_ASSERT(i2 != i);
        size_t begin2, end2;
        neighbor_range(i2, level, &begin2, &end2);
        for (size_t j2 = begin2; j2 < end2; j2++) {
          storage_idx_t i3 = neighbors[j2];
          if (i3 < 0) break;
          if (i3 == i) {
            n_reciprocal++;
            continue;
          }
          if (neighset.count(i3)) {
            neighset.erase(i3);
            n_common++;
          }
        }
      }
      tot_neigh += n_neigh;
      tot_common += n_common;
      tot_reciprocal += n_reciprocal;
    }
  }
  float normalizer = n_node;
  printf("   nb of nodes at that level %ld\n", n_node);
  printf("   neighbors per node: %.2f (%ld)\n",
         tot_neigh / normalizer, tot_neigh);
  printf("   nb of reciprocal neighbors: %.2f\n", tot_reciprocal / normalizer);
  printf("   nb of neighbors that are also neighbor-of-neighbors: %.2f (%ld)\n",
         tot_common / normalizer, tot_common);
}


void HNSW::fill_with_random_links(size_t n)
{
  int max_level = prepare_level_tab(n);
  RandomGenerator rng2(456);

  for (int level = max_level - 1; level >= 0; level++) {
    std::vector<int> elts;
    for (int i = 0; i < n; i++) {
      if (levels[i] > level) {
        elts.push_back(i);
      }
    }
    printf ("linking %ld elements in level %d\n",
            elts.size(), level);

    if (elts.size() == 1) continue;

    for (int ii = 0; ii < elts.size(); ii++) {
      int i = elts[ii];
      size_t begin, end;
      neighbor_range(i, 0, &begin, &end);
      for (size_t j = begin; j < end; j++) {
        int other = 0;
        do {
          other = elts[rng2.rand_int(elts.size())];
        } while(other == i);

        neighbors[j] = other;
      }
    }
  }
}


int HNSW::prepare_level_tab(size_t n, bool preset_levels)
{
  size_t n0 = offsets.size() - 1;

  if (preset_levels) {
    FAISS_ASSERT (n0 + n == levels.size());
  } else {
    FAISS_ASSERT (n0 == levels.size());
    for (int i = 0; i < n; i++) {
      int pt_level = random_level();
      levels.push_back(pt_level + 1);
    }
  }

  int max_level = 0;
  for (int i = 0; i < n; i++) {
    int pt_level = levels[i + n0] - 1;
    if (pt_level > max_level) max_level = pt_level;
    offsets.push_back(offsets.back() +
                      cum_nb_neighbors(pt_level + 1));
    neighbors.resize(offsets.back(), -1);
  }

  return max_level;
}


/// 只保留0层
int HNSW::prepare_level_tab2(size_t n, bool preset_levels) {
/*    // n0 为已添加点的id（0->n0）
    size_t n0 = offsets.size() - 1;
    // printf("offsets.size():%d\n",offsets.size());
    if (preset_levels) {
        FAISS_ASSERT(n0 + n == levels.size());
    } else {
        FAISS_ASSERT(n0 == levels.size());
        for (int i = 0; i < n; i++) {
            // int pt_level = random_level();
            int pt_level = 0;
            levels.push_back(pt_level + 1);
        }
        
        
    }*/
    // 全部位于为0层
    levels.resize(n,1);
    int max_level = 0;
    for (int i = 0; i < n; i++) {
        /*int pt_level = levels[i + n0] - 1;
        if (pt_level > max_level)
            max_level = pt_level;*/
        int pt_level = 0;
        // 记录每个点在neighbors中的位置
        offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
        neighbors.resize(offsets.back(), -1);
    }

    return max_level;
}


// 新索引复制旧索引配置
int HNSW::prepare_level_tab3(HNSW& hnsw1, size_t n, size_t tmp, bool preset_levels) {
  // 复制层次关系
  levels.clear();
  for (size_t i = 0; i < n; i++) {
    levels.push_back(hnsw1.levels[i]);
  }

  int max_level = 0;
  for (size_t i = 0; i < n; i++) {
    // levels中存放的值>=1,因此需要-1计算层次
    int pt_level = levels[i] - 1;
    if (pt_level > max_level) max_level = pt_level;
    // 记录每个点在neighbors中的位置
    offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
    neighbors.resize(offsets.back(), -1);
  }

    // 新申请空间只包含0层
    for (size_t i = n; i < tmp; ++i)
    {
      int pt_level = 0;
      // 记录每个点在neighbors中的位置
      offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
      neighbors.resize(offsets.back(), -1);
    }
  }


// 索引增强(在原始索引后边申请新的空间,层次为0层)
  void HNSW::prepare_level_tab4(idx_t n, idx_t tmp)
{

  // 新申请空间只包含0层，放在索引后边
    for (idx_t i = n; i < tmp; ++i)
    {
      int pt_level = 0;
      // 记录每个点在neighbors中的位置
      offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
      neighbors.resize(offsets.back(), -1);
    }

}


/** Enumerate vertices from farthest to nearest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
// input : 小根堆,存放候选节点，output存放裁后节点
void HNSW::shrink_neighbor_list( 
  DistanceComputer& qdis,
  std::priority_queue<NodeDistFarther>& input,
  std::vector<NodeDistFarther>& output,
  int max_size)
{
  while (input.size() > 0) {
    NodeDistFarther v1 = input.top();
    input.pop();
    float dist_v1_q = v1.d;

    bool good = true;
    // 与output中以存在节点相比较
    for (NodeDistFarther v2 : output) {
      float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);
      // q_v2 为已存在节点，若v1_q距离小于v1_v2距离，则不保存v1作为q的邻居（舍弃三角形较长边）
      if (dist_v1_v2 < dist_v1_q) {
        good = false;
        break;
      }
    }

    if (good) {
      output.push_back(v1);
      if (output.size() >= max_size) {
        return;
      }
    }
  }
}


namespace {


using storage_idx_t = HNSW::storage_idx_t;
using NodeDistCloser = HNSW::NodeDistCloser;
using NodeDistFarther = HNSW::NodeDistFarther;


/**************************************************************
 * Addition subroutines
 **************************************************************/


/// remove neighbors from the list to make it smaller than max_size
void shrink_neighbor_list(
  DistanceComputer& qdis,
  std::priority_queue<NodeDistCloser>& resultSet1,
  int max_size)
{
    if (resultSet1.size() < max_size) {
        return;
    }
    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    // 将大根堆中元素放到小根堆中裁边，结果存入returnlist中
    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    HNSW::shrink_neighbor_list(qdis, resultSet, returnlist, max_size);

    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }

}


/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
void add_link(HNSW& hnsw,
              DistanceComputer& qdis,
              storage_idx_t src, storage_idx_t dest,
              int level)
{
  size_t begin, end;
  hnsw.neighbor_range(src, level, &begin, &end);
  if (hnsw.neighbors[end - 1] == -1) {
    // there is enough room, find a slot to add it
    size_t i = end;
    while(i > begin) {
      if (hnsw.neighbors[i - 1] != -1) break;
      i--;
    }
    hnsw.neighbors[i] = dest;
    return;
  }

  // otherwise we let them fight out which to keep

  // copy to resultSet...
  std::priority_queue<NodeDistCloser> resultSet;
  resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
  for (size_t i = begin; i < end; i++) { // HERE WAS THE BUG
    storage_idx_t neigh = hnsw.neighbors[i];
    resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
  }

  shrink_neighbor_list(qdis, resultSet, end - begin);

  // ...and back
  size_t i = begin;
  while (resultSet.size()) {
    hnsw.neighbors[i++] = resultSet.top().id;
    resultSet.pop();
  }
  // they may have shrunk more than just by 1 element
  while(i < end) {
    hnsw.neighbors[i++] = -1;
  }
}

/// search neighbors on a single level, starting from an entry point
void search_neighbors_to_add(
  HNSW& hnsw,
  DistanceComputer& qdis,
  std::priority_queue<NodeDistCloser>& results,
  int entry_point,
  float d_entry_point,
  int level,
  VisitedTable &vt)
{
  // top is nearest candidate（小根堆）
  std::priority_queue<NodeDistFarther> candidates;

  NodeDistFarther ev(d_entry_point, entry_point);
  candidates.push(ev);
  results.emplace(d_entry_point, entry_point);
  vt.set(entry_point);

  while (!candidates.empty()) {
    // get nearest
    const NodeDistFarther &currEv = candidates.top();

    if (currEv.d > results.top().d) {
      break;
    }
    int currNode = currEv.id;
    candidates.pop();

    // loop over neighbors
    size_t begin, end;
    hnsw.neighbor_range(currNode, level, &begin, &end);
    for(size_t i = begin; i < end; i++) {
      storage_idx_t nodeId = hnsw.neighbors[i];
      if (nodeId < 0) break;
      if (vt.get(nodeId)) continue;
      vt.set(nodeId);

      float dis = qdis(nodeId);
      NodeDistFarther evE1(dis, nodeId);

      if (results.size() < hnsw.efConstruction ||
          results.top().d > dis) {

        results.emplace(dis, nodeId);
        candidates.emplace(dis, nodeId);
        if (results.size() > hnsw.efConstruction) {
          results.pop();
        }
      }
    }
  }
  vt.advance();
}


/**************************************************************
 * Searching subroutines
 **************************************************************/

/// greedily update a nearest vector at a given level
void greedy_update_nearest(const HNSW& hnsw,
                           DistanceComputer& qdis,
                           int level,
                           storage_idx_t& nearest,
                           float& d_nearest)
{
  for(;;) {
    storage_idx_t prev_nearest = nearest;

    size_t begin, end;
    hnsw.neighbor_range(nearest, level, &begin, &end);
    // printf("ok3");
    for(size_t i = begin; i < end; i++) {
      // printf("ok5");
      storage_idx_t v = hnsw.neighbors[i];
      
      if (v < 0) break;
      // storage 存放原始向量为n个
      float dis = qdis(v);
      // printf("ok7\n");
      if (dis < d_nearest) {
        nearest = v;
        d_nearest = dis;
      }
    }
    // printf("ok4");
    if (nearest == prev_nearest) {
      return;
    }
  }
}


/// greedily update a nearest vector at a given level
void greedy_update_nearest2(const HNSW& hnsw,
                           DistanceComputer& qdis,
                           int level,
                           storage_idx_t& nearest,
                           float& d_nearest,idx_t n)
{
  for(;;) {
    storage_idx_t prev_nearest = nearest;

    size_t begin, end;
    hnsw.neighbor_range(nearest, level, &begin, &end);
    // printf("ok3");
    for(size_t i = begin; i < end; i++) {
      // printf("ok5");
      storage_idx_t v = hnsw.neighbors[i];
      
      if (v < 0 || v>=n) break;
      // storage 存放原始向量为n个
      float dis = qdis(v);
      // printf("ok7\n");
      if (dis < d_nearest) {
        nearest = v;
        d_nearest = dis;
      }
    }
    // printf("ok4");
    if (nearest == prev_nearest) {
      return;
    }
  }
}

}  // namespace







/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
/*
  ptdis 距离计算器
  pt_id 当前点id
  nearest 贪心搜索找到的最近邻（ep）
  d_nearest 最近邻的距离
  level 添加的层次
  locks 锁对象数组
  vt 标记数组
*/

void HNSW::add_links_starting_from(DistanceComputer& ptdis,
                                   storage_idx_t pt_id,
                                   storage_idx_t nearest,
                                   float d_nearest,
                                   int level,
                                   omp_lock_t *locks,
                                   VisitedTable &vt)
{
  std::priority_queue<NodeDistCloser> link_targets;

  // 层内贪心搜索，获取efconstruction个候选节点，efconstruction相当于搜索中控制link_target
  search_neighbors_to_add(*this, ptdis, link_targets, nearest, d_nearest,
                          level, vt);

  // but we can afford only this many neighbors
  // M当前层中允许的邻居数
  int M = nb_neighbors(level);

  // todo：建图
  std::priority_queue<NodeDistCloser> temp;
  while(!link_targets.empty()){
    temp.emplace(link_targets.top());
    link_targets.pop();
    auto d = link_targets.top().d;
    auto id = link_targets.top().id;
    if(levels[id]>1)
      globle_cnt++;
    globle_sum++;
  }
  while(!temp.empty()){
    link_targets.emplace(temp.top());
    temp.pop();
  }


  // 裁边
  ::faiss::shrink_neighbor_list(ptdis, link_targets, M);

  std::priority_queue<NodeDistCloser> temp2;
  while(!link_targets.empty()){
    temp2.emplace(link_targets.top());
    link_targets.pop();
    auto d = link_targets.top().d;
    auto id = link_targets.top().id;
    if(levels[id]>1)
      globle_cnt2++;
    globle_sum2++;
  }
  while(!temp2.empty()){
    link_targets.emplace(temp2.top());
    temp2.pop();
  }

  // 添加双向链接
  while (!link_targets.empty()) {
    int other_id = link_targets.top().id;

    omp_set_lock(&locks[other_id]);
    add_link(*this, ptdis, other_id, pt_id, level);
    omp_unset_lock(&locks[other_id]);
    // 添加反向边（如果邻居多了shrink_neighbor_list）
    add_link(*this, ptdis, pt_id, other_id, level);

    link_targets.pop();
  }
}


/**************************************************************
 * Building, parallel
 **************************************************************/

// hnsw 中构图的关键代码
void HNSW::add_with_locks(DistanceComputer& ptdis, int pt_level, int pt_id,
                          std::vector<omp_lock_t>& locks,
                          VisitedTable& vt)
{
  //  greedy search on upper levels

  storage_idx_t nearest;
  // todo :关闭多线程
#pragma omp critical
  {
    nearest = entry_point;


    if (nearest == -1) {
      max_level = pt_level;
      entry_point = pt_id;
    }
  }

  if (nearest < 0) {
    return;
  }

  omp_set_lock(&locks[pt_id]);

  int level = max_level; // level at which we start adding neighbors
  float d_nearest = ptdis(nearest);

  // 上层执行greedy_search（找到进入level的ep）
  for(; level > pt_level; level--) {
    greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
  }

  // 从pt到0层
  for(; level >= 0; level--) {
    add_links_starting_from(ptdis, pt_id, nearest, d_nearest,
                            level, locks.data(), vt);
  }

  omp_unset_lock(&locks[pt_id]);

  if (pt_level > max_level) {
    max_level = pt_level;
    entry_point = pt_id;
  }
}


// 通过unordered_map统计每个结点反向连接个数
void HNSW::hot_hubs_new_neighbors(std::vector<std::unordered_set<idx_t>>& ses,
        std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        idx_t n,int find_neighbors_mode,std::vector<idx_t>& clsf){
  printf("OK1\n");
  if (find_neighbors_mode==0) // 选择热点反向边作为热点新增的候选邻居
  {
    for (size_t i = 0; i < n; ++i)
    {
        size_t begin, end;
        neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end ;j++) {
            int v1 = neighbors[j];
            if(v1<0||v1>n)
                break;
            // 如果v1为k层的热点，该点作为第k层热点v1的候选邻居
            for (int k = 0; k < ses.size(); ++k)
            {
              if(ses[k].find(v1)!=ses[k].end())
                hot_hubs[k][v1].push_back(i);
            }
        }
    }
  } 
  else if (find_neighbors_mode == 1) // 聚类方法寻找邻居
  {
    for (size_t i = 0; i < n; ++i)
    {
        size_t begin, end;
        neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end ;j++) {
            int v1 = neighbors[j];
            if(v1<0||v1>n)
                break;
            // 如果v1为k层的热点并且v1和i属于同一类-->
            // 该点作为第k层热点v1的候选邻居
            for (int k = 0; k < ses.size(); ++k)
            {
              if(ses[k].find(v1)!=ses[k].end() && clsf[i] == clsf[v1])
                hot_hubs[k][v1].push_back(i);
            }
        }
    }
  }
  else if (find_neighbors_mode == 2) // 随机选择邻居
  {
    std::vector<int> vec;
    for (size_t i = 0; i < n; ++i)
    {
        vec.push_back(i);
    }
    
    // 每个热点选择100 个 邻居 不重复
    for (int k = 0; k < ses.size(); ++k) // 第k层
    {
      for (auto a:ses[k]) // 第k层的热点a
      {
        int cur=0;
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(vec.begin(), vec.end(), std::default_random_engine(seed));
        for(int i=0;i<100;i++){ // 为a添加100 个邻居
          hot_hubs[k][a].push_back(vec[cur%n]);
          cur++;
        }
      }
    }
  }   
  printf("OK2\n");
  // todo: find_neighbors_mode (其他热点邻居选择方式)

}




// 子图根据热点选择热点反向边
void HNSW::hot_hubs_new_neighbors_subIndex(std::vector<std::unordered_set<idx_t>>& ses,
        std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        idx_t n,int find_neighbors_mode){
  printf("OK1\n");
  // 选择热点反向边作为热点新增的候选邻居
  if (find_neighbors_mode==0)
  {
    for (size_t i = 0; i < n; ++i)
    {
        // printf("%ld\n",i);
        size_t begin, end;
        neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end ;j++) {
            // printf("neighbors.size():%ld\t,%ld\n",neighbors.size(),j);
            int v1 = neighbors[j];
            if(v1<0||v1>n)
                break;
            // 如果v1为k层的热点，该点作为第k层热点v1的候选邻居
            for (int k = 0; k < ses.size(); ++k)
            {
              if(ses[k].find(v1)!=ses[k].end())
                hot_hubs[k][v1].push_back(i);
            }
        }
    }
  } 
  printf("OK2\n");
}



// 通过unordered_map统计每个结点反向连接个数
void HNSW::find_hot_hubs(std::vector<std::unordered_set<idx_t>>& ses,
    idx_t n, std::vector<float>& ratios){
    std::unordered_map<idx_t,idx_t> ma;
    for (size_t i = 0; i < n; ++i)
    {
        size_t begin, end;
        neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end ;j++) {
            idx_t v1 = neighbors[j];
            if(v1<0||v1>n)
                break;
            ma[v1]++;
            indegrees[v1]++;
        }
    }

    // 频率为first , second:结点编号
    typedef std::pair<int,idx_t> pii;
    std::vector<pii> heat_degrees;
    // 按照热点的热度排序
    for(auto a : ma){
        heat_degrees.push_back(pii(-a.second,a.first));
    }
    std::sort(heat_degrees.begin(),heat_degrees.end());

    // printf("heat_degrees.size():%d\n",heat_degrees.size());
    // 存放不同等级的热点
    idx_t cur=0;
    for (idx_t i = 0; i < ratios.size(); ++i)
    {
      idx_t nb_ratios = n*ratios[i];

      for (idx_t j = cur; j < cur+nb_ratios; ++j)
      {
        // printf("热度: %d\t,热点id:%d\n", heat_degrees[j].first,heat_degrees[j].second);
        ses[i].insert(heat_degrees[j].second);
        // 保存热点及其邻居为 全局变量
        globleHeatDegrees[heat_degrees[j].second]=-heat_degrees[j].first;
      }
      
      cur+=nb_ratios;
    }
}


// 在邻居的邻居范围内寻找热点
void HNSW::find_hot_hubs_with_neighbors(std::vector<std::unordered_set<idx_t>>& ses,
    idx_t n, std::vector<float>& ratios){
    std::unordered_map<idx_t,idx_t> ma;
    for (size_t i = 0; i < n; ++i)
    {
        size_t begin, end;
        neighbor_range(i, 0, &begin, &end);
        // i 的 邻居
        for (size_t j = begin; j < end ;j++) {
            idx_t v1 = neighbors[j];
            if(v1<0||v1>n)
                break;
            size_t begin1, end1;
            neighbor_range(v1, 0, &begin1, &end1);
            // i邻居的邻居：如果i的邻居的邻居指向i，ma[i]++;
            for (size_t ii = begin1; ii < end1 ;ii++) {
                idx_t v2 = neighbors[ii];
                if(v2<0||v2>n)
                    break;
                size_t begin2, end2;
                neighbor_range(v2, 0, &begin2, &end2);
                for (size_t iii = begin2; iii < end2 ;iii++) {
                  idx_t v3 = neighbors[iii];
                  if(v3<0||v3>n)
                      break;
                  if (v3==i)
                  {
                    ma[i]++;    
                  }
                }  
            }  
        }
    }

    // 频率为first , second:结点编号
    typedef std::pair<int,idx_t> pii;
    std::vector<pii> heat_degrees;
    // 按照热点的热度排序
    for(auto a : ma){
        heat_degrees.push_back(pii(-a.second,a.first));
    }
    std::sort(heat_degrees.begin(),heat_degrees.end());

    // printf("heat_degrees.size():%d\n",heat_degrees.size());
    // 存放不同等级的热点
    idx_t cur=0;
    for (idx_t i = 0; i < ratios.size(); ++i)
    {
      idx_t nb_ratios = n*ratios[i];

      for (idx_t j = cur; j < cur+nb_ratios; ++j)
      {
        // printf("热度: %d\t,热点id:%d\n", heat_degrees[j].first,heat_degrees[j].second);
        ses[i].insert(heat_degrees[j].second);
      }
      
      cur+=nb_ratios;
    }
}


// 全局热点与分类热点相似度
void hubs_similarity(std::vector<std::unordered_set<idx_t>>& ses1,
    std::vector<std::unordered_set<idx_t>>& ses2){

  idx_t cnt = 0;
  idx_t sum = 0;
  for (int i = 0; i < ses1.size(); ++i)
  {
    // 获得第一种方法的第i层热点
    auto se1 = ses1[i];
    sum += se1.size();
    for (int j = 0; j < ses2.size(); ++j)
    {
      // 获得第二种方法第j层热点
      auto se2 = ses2[j];
      for (auto a : se2)
      {
        // 说明热点重合
        if (se1.find(a) != se1.end())
        {
          cnt++;
        }
      }
    }
  }
  printf("热点重合个数： %ld, 热点总个数: %ld, 热点重合率 ：%lf \n", cnt,sum,(cnt*1.0)/sum);
}

 
/*
 * 分类方法从每一类中获取邻居固定比例的热点
 * ma 中存放所有点的反向邻居
 * ratios 存放每一层所占的比例
 * clsf 存放每一类的所属类别
 */
void get_hubs_by_ratios(idx_t n, std::unordered_map<idx_t,idx_t>& ma ,
    std::vector<float>& ratios,std::vector<idx_t>& clsf){
    int k = 1;
    // 统计每一类的个数
    std::vector<int> numsofcls(k);
    for (int i = 0; i < n; ++i)
    {
      numsofcls[clsf[i]]++;
    }

/*    for (int i = 0; i < k; ++i)
    {
      printf("第%d结点数目：%d\n",i,numsofcls[i]);
    }*/


        
    // 结点总比例
    float sum_ratios = 0.0;
    for (int i = 0; i < ratios.size(); ++i)
    {
      sum_ratios += ratios[i];
    }

    typedef std::pair<idx_t,idx_t> pii;
    // 每一类的top_cnts 放入priority_queue
    std::vector<std::priority_queue<pii>> vct(k);
    for(auto a : ma){
      // cls：获取类别，cnts：每一类固定比例下对应的点数
      int cls = clsf[a.first];
      int cnts = numsofcls[cls] * sum_ratios ; 
      if(vct[cls].size() < cnts || -a.second < vct[cls].top().first){
        vct[cls].push(pii(-a.second,a.first));
        if (vct[cls].size() > cnts)
        {
          vct[cls].pop();
        }
      }
    }

    // 将固定比例的热点以及反向邻居个数重新放入ma中
    ma.clear();
    for (int i = 0; i < k; ++i)
    {
      while(!vct[i].empty()){
        ma[vct[i].top().second] = -vct[i].top().first ;
        vct[i].pop();
      }
    }
}


// 聚类寻找热点方法，clsf中存放每个点对应类别
void HNSW::find_hot_hubs_with_classfication(std::vector<std::unordered_set<idx_t>>& ses,
    idx_t n, std::vector<float>& ratios,std::vector<idx_t>& clsf){

    // 统计所有结点的反向边个数
    std::unordered_map<idx_t,idx_t> ma;
    for (size_t i = 0; i < n; ++i)
    {
        size_t begin, end;
        neighbor_range(i, 0, &begin, &end);
        // i 的 邻居
        for (size_t j = begin; j < end ;j++) {
            idx_t v1 = neighbors[j];
            if(v1<0||v1>n)
                break;
            // 说明两者属于同一类，也就是说i是j的反向边
            if (clsf[i] == clsf[v1])
            {
              ma[v1]++;
            }
        }
    }

    // 将热点分类，每类取固定的比例然后放入ma中
    get_hubs_by_ratios(n,ma,ratios,clsf);
    printf("ma.size():%d\n",ma.size());

    // 频率为first , second:结点编号
    typedef std::pair<int,idx_t> pii;
    std::vector<pii> heat_degrees;
    // 按照热点的热度排序
    for(auto a : ma){
        heat_degrees.push_back(pii(-a.second,a.first));
    }
    std::sort(heat_degrees.begin(),heat_degrees.end());

    // 热点分层
    idx_t cur=0;
    for (idx_t i = 0; i < ratios.size(); ++i)
    {
      idx_t nb_ratios = n*ratios[i];

      for (idx_t j = cur; j < cur+nb_ratios; ++j)
      {
        ses[i].insert(heat_degrees[j].second);
      }
      
      cur+=nb_ratios;
    }
}

// 将不同层次的热点及其邻居放入hot_hubs
void HNSW::find_hot_hubs_enhence(std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        idx_t n, std::vector<float>& ratios, int find_hubs_mode,
        int find_neighbors_mode,std::vector<idx_t>& clsf){
  
  // 存放寻找的热点
  std::vector<std::unordered_set<idx_t>> ses(ratios.size());

  // 按照入度寻找热点,放入se中
  if(find_hubs_mode==0){
    find_hot_hubs(ses, n, ratios);
  }else if (find_hubs_mode == 1)
  {
    // 根据邻居的邻居方式统计热点
    find_hot_hubs_with_neighbors(ses,n,ratios);
  }else if (find_hubs_mode == 2)
  {
    // 聚类之后，在每类中利用反向邻居个数统计热点
    find_hot_hubs_with_classfication(ses,n,ratios,clsf);    
  }/*else if (find_hubs_mode == 3)
  {
    //两种方式结合的热点选择方法
    std::vector<std::unordered_set<idx_t>> ses1(ratios.size());
    std::vector<std::unordered_set<idx_t>> ses2(ratios.size());
    std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs1;
    std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs2;
    find_hot_hubs(ses1, n, ratios);
    find_hot_hubs_with_classfication(ses2,n,ratios,clsf);
    hot_hubs_new_neighbors(ses1,hot_hubs1,n,0,clsf);
    hot_hubs_new_neighbors(ses2,hot_hubs2,n,1,clsf);
    // 合并两个map，并去重
    // todo : 比例如何分配？ 需要从中选取ratios比例的热点
    // 如何选择
    for(auto a : hot_hubs1){
      ma[a.first] = a.second;
    }

    for(auto a : hot_hubs2){
      if (ma.find(a.first) == ma.end())
      {
        ma[a.first] = a.second;      
      }
    }

  }*/
  // todo : 其他搜索方式

  // 全局与分类相似度计算
/*  std::vector<std::unordered_set<idx_t>> ses1(ratios.size());
  std::vector<std::unordered_set<idx_t>> ses2(ratios.size());
  find_hot_hubs(ses1, n, ratios);
  
  find_hot_hubs_with_classfication(ses2,n,ratios,clsf);
  
  hubs_similarity(ses1,ses2);
*/
  

  // 拿到热点(ses中存放)去寻找热点邻居
  if (find_hubs_mode != 3)
  {
    hot_hubs_new_neighbors(ses,hot_hubs,n,find_neighbors_mode,clsf);
  }
  
/*  printf("hot_hubs.size(): %d\n", hot_hubs.size());
  printf("ses.size() %d\n",ses.size());
  auto se=ses[0];
  for(auto a:se){
    printf("热点：%d\n", a);
  }

  auto ma = hot_hubs[0];
  for(auto a : ma){
    for(auto b:a.second){
      printf("热点反向边 ： %d\n", b);
    }
  }
*/ 
}


// 子图寻找热点及其反向边
void HNSW::find_hot_hubs_enhence_subIndex(std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        idx_t n, std::vector<float>& ratios, int find_hubs_mode,
        int find_neighbors_mode){
  
  // 存放寻找的热点
  std::vector<std::unordered_set<idx_t>> ses(ratios.size());

  // 按照入度寻找热点,放入se中
  if(find_hubs_mode==0){
    find_hot_hubs(ses, n, ratios);
  }

  // 拿到热点(ses中存放)去寻找热点邻居
  hot_hubs_new_neighbors_subIndex(ses,hot_hubs,n,find_neighbors_mode);

  
/*  printf("hot_hubs.size(): %d\n", hot_hubs.size());
  printf("ses.size() %d\n",ses.size());
  auto se=ses[0];
  for(auto a:se){
    printf("热点：%d\n", a);
  }

  auto ma = hot_hubs[0];
  for(auto a : ma){
    for(auto b:a.second){
      printf("热点反向边 ： %d\n", b);
    }
  }
*/ 
}







 
/*
* 为热点添加反向边添加到索引末尾
* 首先，将原始位置填满
* 如果填满将最后一个位置指向索引末尾位置，将剩余邻居填入该位置
*/
void HNSW::fromNearestToFurther(int nb_reverse_nbs_level,std::vector<idx_t>& new_neibor,
                DistanceComputer& dis,idx_t hot_hub){
  std::vector<std::pair<float,idx_t>> v;
  for (auto a : new_neibor)
  {
      // printf("%f\t", dis.symmetric_dis(hot_hub,a));
      v.push_back(std::make_pair(dis.symmetric_dis(hot_hub,a),a));
  }
  // printf("----------------\n");
  std::sort(v.begin(),v.end());
  // std::reverse(v.begin(),v.end());

/*  for (int i = 0; i < v.size(); ++i)
  {
    printf("%f\t", v[i].first );
  }

  printf("----------------\n");*/

  // 0 层分配10个，其他层分配2*M个
  int cnt = nb_reverse_nbs_level;
  cnt = cnt>=10 ? 10 : 2;
  // 0层最大邻居数
  int m = cum_nneighbor_per_level[1];

  // 防止不够new_neibor中不够cnt*m
  new_neibor.resize(std::min((int)new_neibor.size(),cnt*m));

  int nn=v.size();
  std::shuffle(v.begin(), v.begin() + nn/2, std::mt19937{ std::random_device{}() });
  std::shuffle(v.begin()+nn/2, v.begin() + nn, std::mt19937{ std::random_device{}() });
  // 全部放入new_neighbor
  /*for (int i = 0; i < new_neibor.size(); ++i)
  {
      new_neibor[i]=v[i].second;
  }*/

  // todo: 从近距离（前一半）中选择4/5*2 M,从远距离(后一半)中选取1/5*2M;

/*  for (int i = 0; i < nn; ++i)
  {
    printf("%f\t", v[i].first );
  }
  printf("----------------\n");*/

  int cur=0;
  for (int i = 0; i < nn&&cur<new_neibor.size()*4/5; ++i)
  {
    new_neibor[cur++] = v[i].second;
  }

  for (int i = nn/2; i < nn&&cur<new_neibor.size(); ++i)
  {
    new_neibor[cur++] = v[i].second;
  }

  new_neibor.resize(cur);

}


void HNSW::shink_reverse_neighbors(int nb_reverse_nbs_level,std::vector<idx_t>& new_neibor,
                DistanceComputer& dis,idx_t hot_hub,size_t n){
  
  // 将反向邻居按照距离放入小根堆
  std::priority_queue<NodeDistFarther> input;
  std::vector<NodeDistFarther> returnlist;
  std::vector<idx_t> tmp;
  for (auto a : new_neibor)
  {
      // first:与热点距离， second：热点反向邻居
      input.emplace(dis.symmetric_dis(hot_hub,a), (int)a);
      tmp.push_back(a);
  }

  // 0 层分配10个，其他层分配m个
  int cnt = nb_reverse_nbs_level;
  cnt = cnt>=10 ? 10 : 1;
  // 0层最大邻居数
  int m = cum_nneighbor_per_level[1];

  std::unordered_set<idx_t> vis;
  // 将已存在的正向邻居加入
  size_t begin, end;
  neighbor_range(hot_hub, 0, &begin, &end);
  for (size_t i = begin; i < end; i++)
  {
    idx_t v1 = neighbors[i];
    if(v1 < 0||v1 > n)
      break;
    // 类型不一致，NodeDistFarther数据超过int会出问题
    returnlist.push_back(NodeDistFarther(dis.symmetric_dis(hot_hub,v1),(int)v1));
    vis.insert(v1);
  }

  // 原始邻居长度
  int len = returnlist.size();
  HNSW::shrink_neighbor_list(dis, input, returnlist, (cnt+2)*m);
  // 重新组建new_neighbor
  new_neibor.resize(0);
  // 可以放满就放裁边的结果，放不满就放随机邻居
  for (int i = len; i < returnlist.size(); ++i)
  {
    new_neibor.push_back((idx_t)returnlist[i].id);
    vis.insert((idx_t)returnlist[i].id);
  }

  // todo:将new_neibor空位插入随机邻居(加够m)
  for (int i = 0; i < tmp.size()&&new_neibor.size()<cnt*m; ++i)
  {
    if (vis.find(tmp[i])==vis.end())
    {
      vis.insert(tmp[i]);
      new_neibor.push_back(tmp[i]);
    }
  }

}



void HNSW::add_reverse_link_by_indegrees(int nb_reverse_nbs_level,std::vector<idx_t>& new_neibor,
                DistanceComputer& dis,idx_t hot_hub,size_t n){
      // 频率为first , second:结点编号
    int len = new_neibor.size();
    typedef std::pair<int,idx_t> pii;
    std::vector<pii> heat_degrees;
    // 按照热点的热度排序
    for(auto a : new_neibor){
        heat_degrees.push_back(pii(indegrees[a],a));
    }
    std::sort(heat_degrees.begin(),heat_degrees.end());
    for(int i=0;i<len;i++){
        new_neibor[i] = heat_degrees[i].second;
        // std::cout<<heat_degrees[i].first<<std::endl;
    }
}

void HNSW::add_reverse_link_by_indegrees_and_outdegrees(int nb_reverse_nbs_level,std::vector<idx_t>& new_neibor,
                DistanceComputer& dis,idx_t hot_hub,size_t n){
      // 频率为first , second:结点编号
    int len = new_neibor.size();
    std::vector<std::vector<int>> heat_degrees;
    // 按照入度从小到大，出度从大到小排序添加
    for(auto a : new_neibor){
        // 计算当前结点的出度
        int out_degree = 0;
        size_t begin, end;
        neighbor_range(a, 0, &begin, &end);
        // printf("hh.first:%d ,hh.second.size()%d\n",hh.first,hh.second.size());
        // 将该点hh.first邻居放入ma中，防止重复插入
        for (size_t i = begin; i < end; i++)
        {
            idx_t v1 = neighbors[i];
            if(v1 < 0||v1 > n)
                break;
            out_degree++;
        }
        heat_degrees.push_back({indegrees[a],-out_degree,a});
    }
    std::sort(heat_degrees.begin(),heat_degrees.end());
    for(int i=0;i<len;i++){
        new_neibor[i] = heat_degrees[i][2];
    }
}


void caculatePR(idx_t hid,std::vector<idx_t>& new_neibor){
    // 输出热点反向边id
    std::ofstream out("./out_data/globleHubsReverseNbs.csv",std::ofstream::app);
    for(auto a : new_neibor){
        out<<a<<std::endl;
    }
    out.close();



    // 将热点反向边添加到新图中
    std::ofstream out1("./out_data/gWithHots.csv",std::ofstream::app);
    for(auto a:new_neibor){
        out1<<hid<<",";
        out1<<a<<std::endl;
    }
    out1.close();  

}



void HNSW::add_new_reverse_link_end_enhence(
        std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        size_t n,DistanceComputer& dis,std::vector<int>& nb_reverse_neighbors){
        
        // 反向邻居入度
        std::map<int,int> reverseNbsIndegrees;

        size_t cur_pos=n;
        int nb_level = hot_hubs.size();
        for (int ii = 0; ii < nb_level; ii++)
        {
          // 第ii层的热点及其邻居，类型：unordered_map<idx_t,std::vector<idx_t>>
          auto each_level = hot_hubs[ii];
          for(auto hh : each_level){
              // 找到该热点，将反向边插入到后续位置上
              size_t begin, end;
              neighbor_range(hh.first, 0, &begin, &end);
              // printf("hh.first:%d ,hh.second.size()%d\n",hh.first,hh.second.size());
              // 将该点hh.first邻居放入ma中，防止重复插入
              std::unordered_set<idx_t> se;
              for (size_t i = begin; i < end; i++)
              {
                  idx_t v1 = neighbors[i];
                  if(v1 < 0||v1 > n)
                      break;
                  se.insert(v1);
              }

              auto new_neibor = hh.second;
              // todo：按照距离排序
              // fromNearestToFurther(nb_reverse_neighbors[ii],new_neibor,dis,hh.first);
              // todo : 热点反向边裁边
              // shink_reverse_neighbors(nb_reverse_neighbors[ii],new_neibor,dis,hh.first,n);
              // todo : 按照入度较小排序添加反向边
              // add_reverse_link_by_indegrees(nb_reverse_neighbors[ii],new_neibor,dis,hh.first,n);
              add_reverse_link_by_indegrees_and_outdegrees(nb_reverse_neighbors[ii],new_neibor,dis,hh.first,n);
              int m = cum_nneighbor_per_level[1];
              // 统计需要添加的反向邻居的入度分布
              int pp = 0;
              while (pp < new_neibor.size() && pp<nb_reverse_neighbors[ii]*(m-1))
              {
                  idx_t curNode = new_neibor[pp++];
                  reverseNbsIndegrees[indegrees[curNode]]++;
              }
              // 将该节点的新邻居输出到新构图，计算PR
              // caculatePR(hh.first,new_neibor);
              // 指向热点邻居的指针
              int p=0;
              // 将新邻居插入到后边,添加cnt个邻居（5，10，15，20，25）
              int cnt=0;
              for (size_t j = begin; 
                  j < end && p < new_neibor.size() /*&& cnt < 15*/;) {
                  int v1 = neighbors[j];
                  // 如果该邻居已经存在，就不用重复添加该邻居
                  if(se.find(new_neibor[p])!=se.end()){
                      p++;
                      continue;
                  }
                  if(v1 < 0)
                      neighbors[j]=new_neibor[p++];
                  j++;
              }

              size_t pre=end;
              // 旧位置全部占用，但是热点邻居还没有放置完毕
              // m每个位置只能分配m-1个数据，一个位置存放指针
              while (p < new_neibor.size() && p<nb_reverse_neighbors[ii]*(m-1))
              {
                  // 新位置所处空间位置
                  size_t begin1, end1;
                  neighbor_range(cur_pos, 0, &begin1, &end1);
                  // 将新位置的第一个元素，存放旧位置最后位置的元素
                  // printf("cur_pos :%d,end-1: %d\n",cur_pos,hnsw.neighbors[end-1]);
                  neighbors[begin1] = neighbors[pre-1]; 

                  /*printf("%d\t,%d\n",cur_pos,hnsw.neighbors[end-1])*/
                  // 就位置的末尾指向新位置
                  neighbors[pre-1] = cur_pos;

                  cur_pos++;
                  // 新位置都是-1
                  for (size_t k = begin1+1;
                        k < end1 && p < new_neibor.size()&& p<nb_reverse_neighbors[ii]*(m-1);)
                  {
                      neighbors[k++]=new_neibor[p++];
                  }
                  pre=end1;
              }
          }
      }

    std::ofstream out("./reverseNbsIndegrees.csv",std::ofstream::app);

    for(auto a:reverseNbsIndegrees){
        out<<a.first<<",";
        out<<a.second<<std::endl;
    }
    out<<std::endl;
    out.close();
 

}


// 将热点的邻居连向热点
void HNSW::add_links_to_hubs(
        std::vector<std::unordered_map<idx_t,std::vector<idx_t>>>& hot_hubs,
        size_t n){

  // 热点层次遍历
  for (int l = 0; l < hot_hubs.size(); ++l)
  {
      // 第i层热点遍历
      for(auto& hub : hot_hubs[l]){
        // 热点id
        idx_t hubID = hub.first;
        std:std::vector<idx_t> nbs = hub.second;

        // 遍历热点邻居连向热点
        for (int i = 0; i < nbs.size(); ++i)
        {
          idx_t cur = nbs[i];
          // 邻居的邻居是否已经满 ？ 满去掉最后一个连接，不满直接添加到后边
          size_t begin,end;
          neighbor_range(cur,0,&begin,&end);
          // 不满
          if (neighbors[end-1]==-1)
          {
            for (size_t i = begin; i < end; ++i)
            {
              if (neighbors[i] == -1)
              {
                neighbors[i] = hubID;
                break;
              }
            }
          }else {
            neighbors[end-1] = hubID;
          }
        }
      }
  }

}



/** Do a BFS on the candidates list */

int HNSW::search_from_candidates(
  DistanceComputer& qdis, int k,
  idx_t *I, float *D,
  MinimaxHeap& candidates,
  VisitedTable& vt,
  int level, int nres_in) const
{
  int nres = nres_in;
  int ndis = 0;

  for (int i = 0; i < candidates.size(); i++) {
    idx_t v1 = candidates.ids[i];
    float d = candidates.dis[i];
    FAISS_ASSERT(v1 >= 0);
    if (nres < k) {
      faiss::maxheap_push(++nres, D, I, d, v1);
    } else if (d < D[0]) {
      faiss::maxheap_pop(nres--, D, I);
      faiss::maxheap_push(++nres, D, I, d, v1);
    }
    vt.set(v1);
  }

  int nstep = 0;

  while (candidates.size() > 0) {
    float d0 = 0;

    int v0 = candidates.pop_min(&d0);

    size_t begin, end;
    neighbor_range(v0, level, &begin, &end);

    for (size_t j = begin; j < end; j++) {
      int v1 = neighbors[j];
      if (v1 < 0) break;
      if (vt.get(v1)) {
        continue;
      }
      vt.set(v1);
      ndis++;
      float d = qdis(v1);
      if (nres < k) {
        faiss::maxheap_push(++nres, D, I, d, v1);
      } else if (d < D[0]) {
        faiss::maxheap_pop(nres--, D, I);
        faiss::maxheap_push(++nres, D, I, d, v1);
      }
      candidates.push(v1, d);
    }

    nstep++;
    // efsearch 控制搜索结束
    if (nstep > efSearch) {
      break;
    }
  }

  if (level == 0) {
// #pragma omp critical
    {
      hnsw_stats.n1 ++;
      if (candidates.size() == 0) {
        hnsw_stats.n2 ++;
      }
      hnsw_stats.n3 += ndis;
    }
  }
  faiss::maxheap_reorder (k, D, I);
  D[0]=ndis;
  D[1]=nstep;

  return nres;
}



/*
* 在search_from_candidates的基础上，修改了候选队列的数据结构
* 由原本的线性扫描换为优先队列
*/
/*
template<typename T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;
  // 小根堆，存放候选节点

int HNSW::search_from_candidates_optimize(
  DistanceComputer& qdis, int k,
  idx_t *I, float *D,
  MinimaxHeap& candidates,
  VisitedTable& vt,
  int level, int nres_in) const
{
  int nres = nres_in;
  int ndis = 0;

  float d0 = 0;
  int v0 = candidates.pop_min(&d0);

  MinHeap<Node> candidate_set;
  candidate_set.emplace(d0, v0);

  int nstep = 0;

  while (candidate_set.size() > 0) {
    float d0;
    storage_idx_t v0;
    std::tie(d0, v0) = candidate_set.top();

    candidate_set.pop();
    
    size_t begin, end;
    neighbor_range(v0, level, &begin, &end);

    for (size_t j = begin; j < end; j++) {
      int v1 = neighbors[j];
      if (v1 < 0) break;
      if (vt.get(v1)) {
        continue;
      }
      vt.set(v1);
      ndis++;
      float d = qdis(v1);
      if (nres < k) {
        faiss::maxheap_push(++nres, D, I, d, v1);
      } else if (d < D[0]) {
        faiss::maxheap_pop(nres--, D, I);
        faiss::maxheap_push(++nres, D, I, d, v1);
      }
      candidate_set.emplace(d, v1);
    }

    nstep++;
    // efsearch 控制搜索结束
    if (nstep > efSearch) {
      break;
    }
  }

  if (level == 0) {
// #pragma omp critical
    {
      hnsw_stats.n1 ++;
      if (candidates.size() == 0) {
        hnsw_stats.n2 ++;
      }
      hnsw_stats.n3 += ndis;
    }
  }

  D[0]=ndis;
  D[1]=nstep;

  return nres;
}
*/
/**************************************************************
 * Searching
 **************************************************************/

template<typename T>
using MaxHeap = std::priority_queue<T, std::vector<T>, std::less<T>>;
template<typename T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;

MaxHeap<HNSW::Node> HNSW::search_from(
  const Node& node,
  DistanceComputer& qdis,
  int ef,
  VisitedTable *vt) const
{
  // 大根堆，存放结果
  MaxHeap<Node> top_candidates;
  // 小根堆，存放候选节点
  MinHeap<Node> candidate_set;

  top_candidates.push(node);
  candidate_set.push(node);

  vt->set(node.second);

  float lower_bound = node.first;

  // 记录访问点的个数
  storage_idx_t ndis=0;

  while (!candidate_set.empty()) {
    float d0;
    storage_idx_t v0;
    std::tie(d0, v0) = candidate_set.top();

    // 当前候选中最小，大于结果集中的最大
    if (d0 > lower_bound) {
      break;
    }

    candidate_set.pop();

    size_t begin, end;
    neighbor_range(v0, 0, &begin, &end);

    for (size_t j = begin; j < end; ++j) {
      int v1 = neighbors[j];

      if (v1 < 0) {
        break;
      }
      if (vt->get(v1)) {
        continue;
      }

      vt->set(v1);

      float d1 = qdis(v1);
      ndis++;
      if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
        candidate_set.emplace(d1, v1);
        top_candidates.emplace(d1, v1);

        if (top_candidates.size() > ef) {
          top_candidates.pop();
        }
        // 更新结果集中的最大
        lower_bound = top_candidates.top().first;
      }

    }
  }
  // 传递访问点的个数：寻找当前队列中的距离最大值，然后push(lower_bound+1,ndis)
  // 如果top_candidates为空会报错
  FAISS_THROW_IF_NOT_MSG(!top_candidates.empty()," top_candidates is empty！");
  lower_bound = top_candidates.top().first;
  top_candidates.emplace(lower_bound+1,ndis);
  return top_candidates;
}


template<typename T>
using MaxHeap = std::priority_queue<T, std::vector<T>, std::less<T>>;
template<typename T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;

// 返回访问点的个数
void HNSW::search_from_hubs_to_nbs(
  const Node& node,
  idx_t q,
  DistanceComputer& qdis,
  int ef,
  VisitedTable& vt,std::map<int,int>& stepsM) const
{

  // printf("%ld\t,%ld\n",node.second,q);  
  // 大根堆，存放结果
  MaxHeap<Node> top_candidates;
  // 小根堆，存放候选节点
  MinHeap<Node> candidate_set;

  // 放入起始点
  top_candidates.push(node);
  candidate_set.push(node);

  vt.set(node.second);

  int nstep = 0;
  int flag = 0; // 记录是否找到 
  int flag2 = 0;
  while (!candidate_set.empty()) {
    float d0;
    idx_t v0;
    std::tie(d0, v0) = candidate_set.top();

    candidate_set.pop();
    
    nstep++;

    if(nstep > 200){
      stepsM[200]++;
      flag2=1;
      break;
    }

    size_t begin, end;
    neighbor_range(v0, 0, &begin, &end);

    for (size_t j = begin; j < end; ++j) {
      idx_t v1 = neighbors[j];

      if (v1==q) // 找到该点
      {
        stepsM[nstep]++;
        flag = 1;
        flag2 = 1;
        break;
      }

      if (v1 < 0) {
        break;
      }

      if (vt.get(v1)) {
        continue;
      }

      vt.set(v1);

      float d1 = qdis.symmetric_dis(v1,q);
      // printf("%f\n",d1);

      if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
        candidate_set.emplace(d1, v1);
        top_candidates.emplace(d1, v1);

        if (top_candidates.size() > ef) {
          top_candidates.pop();
        }
      }
    }
    if (flag)
      break;
  }

  if (!flag2)
  {
    stepsM[10001]++;
  }

}


// 在search_from 的基础上解决k对unbounded的影响

template<typename T>
using MaxHeap = std::priority_queue<T, std::vector<T>, std::less<T>>;
template<typename T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;

MaxHeap<HNSW::Node> HNSW::search_from_addk(
  const Node& node,
  DistanceComputer& qdis,
  int ef,
  VisitedTable *vt ,int k) const
{
  // 大根堆，存放结果
  MaxHeap<Node> top_candidates;
  // 小根堆，存放候选节点
  MinHeap<Node> candidate_set;
  // 大根堆存放返回结果
  MaxHeap<Node> results;

  top_candidates.push(node);
  candidate_set.push(node);
  results.push(node);

  vt->set(node.second);

  float lower_bound = node.first;

  // 记录访问点的个数
  storage_idx_t ndis=0;

  while (!candidate_set.empty()) {
    float d0;
    storage_idx_t v0;
    std::tie(d0, v0) = candidate_set.top();

    // 当前候选中最小，大于结果集中的最大
    if (d0 > lower_bound) {
      break;
    }

    candidate_set.pop();

    size_t begin, end;
    neighbor_range(v0, 0, &begin, &end);

    for (size_t j = begin; j < end; ++j) {
      int v1 = neighbors[j];

      if (v1 < 0) {
        break;
      }
      if (vt->get(v1)) {
        continue;
      }

      vt->set(v1);

      float d1 = qdis(v1);
      ndis++;
      if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
        candidate_set.emplace(d1, v1);
        top_candidates.emplace(d1, v1);
        if (top_candidates.size() > ef) {
            top_candidates.pop();
        }

        // 将放入top_candidates中的数据，放到result中存储（目的：top_candidates大小不变的情况下，还能取到k个值）。
        if (results.top().first > d1 || results.size() < k)
            results.emplace(d1,v1);

        if(results.size()>k)
            results.pop();

        lower_bound = top_candidates.top().first;
      }

    }
  }
  // 传递访问点的个数：寻找当前队列中的距离最大值，然后push(lower_bound+1,ndis)
  // 如果top_candidates为空会报错
  FAISS_THROW_IF_NOT_MSG(!top_candidates.empty()," top_candidates is empty！");
  lower_bound = results.top().first;
  results.emplace(lower_bound+1,ndis);
  return results;
}



// 在search_from 的基础上解决k对unbounded的影响，通过set优化
// 用unordered_set存储top_candidate中元素，然后将candidate中元素和candidate_set_pop元素添加到top_candidate中

template<typename T>
using MaxHeap = std::priority_queue<T, std::vector<T>, std::less<T>>;
template<typename T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;

MaxHeap<HNSW::Node> HNSW::search_from_addk_v2(
  const Node& node,
  DistanceComputer& qdis,
  int ef,
  VisitedTable *vt ,int k) const
{
  // 大根堆，存放结果
  MaxHeap<Node> top_candidates;
  // 小根堆，存放候选节点
  MinHeap<Node> candidate_set;
  //存放candidate_set弹出元素
  std::vector<Node> candidate_set_pop;

  std::unordered_set<idx_t> top_candidates_set;

  top_candidates.push(node);
  candidate_set.push(node);
  top_candidates_set.emplace(node.second);

  vt->set(node.second);

  float lower_bound = node.first;

  // 记录访问点的个数
  storage_idx_t ndis=0;

  while (!candidate_set.empty()) {
    float d0;
    storage_idx_t v0;
    std::tie(d0, v0) = candidate_set.top();

    // 将candidate_set弹出元素暂存
    candidate_set_pop.push_back(candidate_set.top());
    // 当前候选中最小，大于结果集中的最大
    if (d0 > lower_bound) {
      break;
    }

    candidate_set.pop();

    size_t begin, end;
    neighbor_range(v0, 0, &begin, &end);

    for (size_t j = begin; j < end; ++j) {
      int v1 = neighbors[j];

      if (v1 < 0) {
        break;
      }
      if (vt->get(v1)) {
        continue;
      }

      vt->set(v1);

      float d1 = qdis(v1);
      ndis++;
      if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
        candidate_set.emplace(d1, v1);
        top_candidates.emplace(d1, v1);
        top_candidates_set.emplace(v1);

        if (top_candidates.size() > ef) {
            candidate_set_pop.push_back(top_candidates.top());
            top_candidates_set.erase(top_candidates.top().second);
            top_candidates.pop();
        }

        lower_bound = top_candidates.top().first;
      }

    }
  }


  //first : add to top_candidate
  int n = candidate_set_pop.size();
  for (int i = 0; i < n ; ++i)
  {
      if (!top_candidates_set.count(candidate_set_pop[i].second))
      {
          /*top_candidates.emplace(candidate_set_pop[i]);
          top_candidates_set.emplace(candidate_set_pop[i].second);*/
          candidate_set.emplace(candidate_set_pop[i]);
      }
      
  }

  // second : add to top_candidate
  while(!candidate_set.empty() && top_candidates.size()<k){
      if (!top_candidates_set.count(candidate_set.top().second))
      {
          top_candidates.emplace(candidate_set.top());
          top_candidates_set.emplace(candidate_set.top().second);
      }
      candidate_set.pop();
  }

  lower_bound = top_candidates.top().first;

  top_candidates.emplace(lower_bound+1,ndis);

  return top_candidates;
}



// 统计找到最近邻时访问点个数
template<typename T>
using MaxHeap = std::priority_queue<T, std::vector<T>, std::less<T>>;
template<typename T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;

MaxHeap<HNSW::Node> HNSW::search_from_find_ndis(
  const Node& node,
  DistanceComputer& qdis,
  int ef,
  VisitedTable *vt,std::vector<int>& ndiss,idx_t gt) const
{
  // 大根堆，存放结果
  MaxHeap<Node> top_candidates;
  // 小根堆，存放候选节点
  MinHeap<Node> candidate_set;

  top_candidates.push(node);
  candidate_set.push(node);

  vt->set(node.second);

  float lower_bound = node.first;

  // 记录访问点的个数
  storage_idx_t ndis=0;

  int flag=0;
  while (!candidate_set.empty()) {
    float d0;
    storage_idx_t v0;
    std::tie(d0, v0) = candidate_set.top();

    if(v0==gt){
        flag=1;
        break;
    }

    // 当前候选中最小，大于结果集中的最大
    if (d0 > lower_bound) {
      break;
    }

    candidate_set.pop();

    size_t begin, end;
    neighbor_range(v0, 0, &begin, &end);

    for (size_t j = begin; j < end; ++j) {
      int v1 = neighbors[j];

      if (v1 < 0) {
        break;
      }
      if (vt->get(v1)) {
        continue;
      }

      vt->set(v1);

      float d1 = qdis(v1);
      ndis++;

      if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
        candidate_set.emplace(d1, v1);
        top_candidates.emplace(d1, v1);

        if (top_candidates.size() > ef) {
          top_candidates.pop();
        }
        // 更新结果集中的最大
        lower_bound = top_candidates.top().first;
      }
    }
  }
  if (flag==0)
    ndiss.push_back(-1);
  else
    ndiss.push_back(ndis);
  // 传递访问点的个数：寻找当前队列中的距离最大值，然后push(lower_bound+1,ndis)
  // 如果top_candidates为空会报错
  FAISS_THROW_IF_NOT_MSG(!top_candidates.empty()," top_candidates is empty！");
  lower_bound = top_candidates.top().first;
  top_candidates.emplace(lower_bound+1,ndis);
  return top_candidates;
}





// upper_beam == 1 greedy_update_nearest寻找初始节点
// upper_beam ！= 1 search_from_candidate寻找多个初始节点
void HNSW::search(DistanceComputer& qdis, int k,
                  idx_t *I, float *D,
                  VisitedTable& vt) const
{
  if (upper_beam == 1) {

    //  greedy search on upper levels
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for(int level = max_level; level >= 1; level--) {
      greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
    }

    int ef = std::max(efSearch, 10);

    // search_mode==0 标准的unbounded；search_mode==1，添加k属性的unbounded
    
    MaxHeap<Node> top_candidates = search_from(Node(d_nearest, nearest), qdis, ef, &vt);

    // MaxHeap<Node> top_candidates = search_from_addk(Node(d_nearest, nearest), qdis, ef, &vt,k);
    // 将第一个元素取出


    float d0;
    storage_idx_t ndis;
    std::tie(d0, ndis) = top_candidates.top();
    top_candidates.pop();

    while (top_candidates.size() > k) {
      top_candidates.pop();
    }

    int nres = 0;
    while (!top_candidates.empty()) {
      float d;
      storage_idx_t label;
      std::tie(d, label) = top_candidates.top();
      faiss::maxheap_push(++nres, D, I, d, label);
      top_candidates.pop();
    }
    // 结果重排序
    faiss::maxheap_reorder (k, D, I);
    D[0]=ndis;
    vt.advance();

  } else {

    assert(false);

    int candidates_size = upper_beam;
    MinimaxHeap candidates(candidates_size);

    std::vector<idx_t> I_to_next(candidates_size);
    std::vector<float> D_to_next(candidates_size);

    int nres = 1;
    I_to_next[0] = entry_point;
    D_to_next[0] = qdis(entry_point);

    for(int level = max_level; level >= 0; level--) {

      // copy I, D -> candidates

      candidates.clear();

      for (int i = 0; i < nres; i++) {
        candidates.push(I_to_next[i], D_to_next[i]);
      }

      if (level == 0) {
        nres = search_from_candidates(qdis, k, I, D, candidates, vt, 0);
      } else  {
        nres = search_from_candidates(
          qdis, candidates_size,
          I_to_next.data(), D_to_next.data(),
          candidates, vt, level
        );
      }
      vt.advance();
    }
  }

}




template<typename T>
using MaxHeap = std::priority_queue<T, std::vector<T>, std::less<T>>;
template<typename T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;

MaxHeap<HNSW::Node> HNSW::search_from_add_vct(
    const Node& node,
    DistanceComputer& qdis,
    int ef,
    VisitedTable *vt,std::vector<idx_t> &vct) const
{
  // 大根堆，存放结果
  MaxHeap<Node> top_candidates;
  // 小根堆，存放候选节点
  MinHeap<Node> candidate_set;

  top_candidates.push(node);
  candidate_set.push(node);

  vt->set(node.second);

  float lower_bound = node.first;

  // 记录访问点的个数
  storage_idx_t ndis=0;

  while (!candidate_set.empty()) {
    float d0;
    storage_idx_t v0;
    std::tie(d0, v0) = candidate_set.top();

    // 当前候选中最小，大于结果集中的最大
    if (d0 > lower_bound) {
      break;
    }

    candidate_set.pop();

    size_t begin, end;
    neighbor_range(v0, 0, &begin, &end);

    for (size_t j = begin; j < end; ++j) {
      int v1 = neighbors[j];

      if (v1 < 0) {
        break;
      }
      if (vt->get(v1)) {
        continue;
      }

      vt->set(v1);

      // 访问点记录
      vct.push_back(v1);

      float d1 = qdis(v1);
      ndis++;
      if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
        candidate_set.emplace(d1, v1);
        top_candidates.emplace(d1, v1);

        if (top_candidates.size() > ef) {
          top_candidates.pop();
        }
        // 更新结果集中的最大
        lower_bound = top_candidates.top().first;
      }

    }
  }
  // 传递访问点的个数：寻找当前队列中的距离最大值，然后push(lower_bound+1,ndis)
  // 如果top_candidates为空会报错
  FAISS_THROW_IF_NOT_MSG(!top_candidates.empty()," top_candidates is empty！");
  lower_bound = top_candidates.top().first;
  top_candidates.emplace(lower_bound+1,ndis);
  return top_candidates;
}


// upper_beam == 1 greedy_update_nearest寻找初始节点
// upper_beam ！= 1 search_from_candidate寻找多个初始节点
void HNSW::search_rt_array(DistanceComputer& qdis, int k,
                  idx_t *I, float *D,
                  VisitedTable& vt,std::vector<idx_t> &vct) const
{
  if (upper_beam == 1) {

    //  greedy search on upper levels
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for(int level = max_level; level >= 1; level--) {
      greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
    }

    int ef = std::max(efSearch, 10);

    // search_mode==0 标准的unbounded；search_mode==1，添加k属性的unbounded
    
    MaxHeap<Node> top_candidates = search_from_add_vct(Node(d_nearest, nearest), qdis, ef, &vt,vct);

    // MaxHeap<Node> top_candidates = search_from_addk(Node(d_nearest, nearest), qdis, ef, &vt,k);
    // 将第一个元素(记录访问点信息)取出
    top_candidates.pop();
    while (top_candidates.size() > k) {
      top_candidates.pop();
    }
    int nres = 0;
    while (!top_candidates.empty()) {
      float d;
      storage_idx_t label;
      std::tie(d, label) = top_candidates.top();
      faiss::maxheap_push(++nres, D, I, d, label);
      top_candidates.pop();
    }
    // 结果重排序
    faiss::maxheap_reorder (k, D, I);
    vt.advance();

  }

}



/*

// 修改的search
// upper_beam == 1 greedy_update_nearest寻找初始节点
// upper_beam ！= 1 search_from_candidate寻找多个初始节点
void HNSW::search(DistanceComputer& qdis, int k,
                  idx_t *I, float *D,
                  VisitedTable& vt) const
{
    HNSWStats stats;
    // 上层传过来最近邻节点为1个
    if (upper_beam == 1) {
        //  greedy search on upper levels
        storage_idx_t nearest = entry_point;
        float d_nearest = qdis(nearest);

        for (int level = max_level; level >= 1; level--) {
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        }

        int ef = std::max(efSearch, k);

        // candidates为efsearch k最大值
        MinimaxHeap candidates(ef);

        candidates.push(nearest, d_nearest);

        search_from_candidates(qdis, k, I, D, candidates, vt, 0);

        vt.advance();
  } 

}
*/


// upper_beam == 1 greedy_update_nearest寻找初始节点
// upper_beam ！= 1 search_from_candidate寻找多个初始节点
void HNSW::search_custom(DistanceComputer& qdis, int k,
                  idx_t *I, float *D,
                  VisitedTable& vt,int search_mode) const
{
  if (upper_beam == 1) {

    //  greedy search on upper levels
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for(int level = max_level; level >= 1; level--) {
      greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
    }

    int ef = std::max(efSearch, 10);

    // search_mode==0 标准的unbounded；search_mode==1，添加k属性的unbounded
    MaxHeap<Node> top_candidates;
    if(search_mode == 0)
        top_candidates = search_from(Node(d_nearest, nearest), qdis, ef, &vt);
    else if (search_mode == 1)
    {
      top_candidates = search_from_addk_v2(Node(d_nearest, nearest), qdis, ef, &vt,k);
    }
    else if (search_mode == 2)
    {
      //top_candidates = search_from_two_index(qdis, ef, &vt,k);
    }
        
    // 将第一个元素取出
    float d0;
    storage_idx_t ndis;
    std::tie(d0, ndis) = top_candidates.top();
    top_candidates.pop();

    while (top_candidates.size() > k) {
      top_candidates.pop();
    }


    int nres = 0;
    while (!top_candidates.empty()) {
      float d;
      storage_idx_t label;
      std::tie(d, label) = top_candidates.top();
      faiss::maxheap_push(++nres, D, I, d, label);
      top_candidates.pop();
    }
    // 结果重排序
    faiss::maxheap_reorder (k, D, I);
    D[0]=ndis;
    vt.advance();

  } else {

    assert(false);

    int candidates_size = upper_beam;
    MinimaxHeap candidates(candidates_size);

    std::vector<idx_t> I_to_next(candidates_size);
    std::vector<float> D_to_next(candidates_size);

    int nres = 1;
    I_to_next[0] = entry_point;
    D_to_next[0] = qdis(entry_point);

    for(int level = max_level; level >= 0; level--) {

      // copy I, D -> candidates

      candidates.clear();

      for (int i = 0; i < nres; i++) {
        candidates.push(I_to_next[i], D_to_next[i]);
      }

      if (level == 0) {
        nres = search_from_candidates(qdis, k, I, D, candidates, vt, 0);
      } else  {
        nres = search_from_candidates(
          qdis, candidates_size,
          I_to_next.data(), D_to_next.data(),
          candidates, vt, level
        );
      }
      vt.advance();
    }
  }

}

/** Do a BFS on the candidates list */
/** Do a BFS on the candidates list */

void HNSW::search_from_candidates_combine(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        int level,
        int nres_in,int fos) const {
  int nres = nres_in;
  int ndis = 0;

  for (int i = 0; i < candidates.size(); i++) {
    idx_t v1 = candidates.ids[i];
    float d = candidates.dis[i];
    FAISS_ASSERT(v1 >= 0);
    if (nres < k) {
      faiss::maxheap_push(++nres, D, I, d, v1);
    } else if (d < D[0]) {
      faiss::maxheap_pop(nres--, D, I);
      faiss::maxheap_push(++nres, D, I, d, v1);
    }
    vt.set(v1);
  }

  int nstep = 0;

  while (candidates.size() > 0) {
    float d0 = 0;
    int v0 = candidates.pop_min(&d0);

    size_t begin, end;
    neighbor_range(v0, level, &begin, &end);
    if (!fos)
    {  
        for (size_t j = begin; j < end; j++) {
            int v1 = neighbors[j];
            if (v1 < 0)
                break;
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);
            ndis++;
            float d = qdis(v1);

            if (nres < k) {
              faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
              faiss::maxheap_pop(nres--, D, I);
              faiss::maxheap_push(++nres, D, I, d, v1);
            }
            candidates.push(v1, d);
        }
    }else {
        for (size_t j = end-1; j >=begin; j--) {
            int v1 = neighbors[j];
            if (v1 < 0)
                break;
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);
            ndis++;
            float d = qdis(v1);
            if (nres < k) {
              faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
              faiss::maxheap_pop(nres--, D, I);
              faiss::maxheap_push(++nres, D, I, d, v1);
            }
            candidates.push(v1, d);
        }
    } // end else
    nstep++;
    if (nstep > efSearch) {
        break;
    }
    } // end while

  while (candidates.size() > 0) {
    float d0 = 0;

    int v0 = candidates.pop_min(&d0);

    size_t begin, end;
    neighbor_range(v0, level, &begin, &end);

    for (size_t j = begin; j < end; j++) {
      int v1 = neighbors[j];
      if (v1 < 0) break;
      if (vt.get(v1)) {
        continue;
      }
      vt.set(v1);
      ndis++;
      float d = qdis(v1);
      if (nres < k) {
        faiss::maxheap_push(++nres, D, I, d, v1);
      } else if (d < D[0]) {
        faiss::maxheap_pop(nres--, D, I);
        faiss::maxheap_push(++nres, D, I, d, v1);
      }
      candidates.push(v1, d);
    }

    nstep++;
    // efsearch 控制搜索结束
    if (nstep > efSearch) {
      break;
    }
  }

  faiss::maxheap_reorder (k, D, I);
}

/// standard unbounded search
std::priority_queue<HNSW::Node> HNSW::search_from_candidate_unbounded_combine(
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        int fos) const {
    int ndis = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;

    top_candidates.push(node);
    candidates.push(node);
    vt->set(node.second);
    size_t cnt=0;
    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }

        candidates.pop();
        size_t begin, end;
        neighbor_range(v0, 0, &begin, &end);
        if (fos==0)
        {
            for (size_t j = begin; j < end; j++) {
                int v1 = neighbors[j];
                if (v1 < 0 ) {
                    break;
                }
                if (vt->get(v1)) {
                    continue;
                }
                vt->set(v1);
                float d1 = qdis(v1);
                ++ndis;

                if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                    candidates.emplace(d1, v1);
                    top_candidates.emplace(d1, v1);
                    if (top_candidates.size() > ef) {
                        top_candidates.pop();
                    }
                }
            }
        }else {
            for (size_t j = end-1; j >=begin; j--) {
                int v1 = neighbors[j];

                if (v1 < 0) {
                    break;
                }
                if (vt->get(v1)) {
                    continue;
                }

                vt->set(v1);

                float d1 = qdis(v1);
                ++ndis;
                if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                    candidates.emplace(d1, v1);
                    top_candidates.emplace(d1, v1);
                    if (top_candidates.size() > ef) {
                        top_candidates.pop();
                    }
                }
            }
        } // end else
        
    }// end while

    return top_candidates;
}

void HNSW::combine_search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        int fos,
        VisitedTable& vt,
        RandomGenerator& rng3) const {
    if (upper_beam == 1) {
        //  greedy search on upper levels 
        storage_idx_t nearest = rng3.rand_long()%levels.size();
        // printf("nearest : %ld\n", nearest );
        float d_nearest = qdis(nearest);

/*        for (int level = max_level; level >= 1; level--) {
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        }*/

        int ef = std::max(efSearch, 10);

        if (search_bounded_queue) {
            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);

            search_from_candidates_combine(qdis, k, I, D, candidates, 
                vt, 0, 0 ,fos);
        } else {
            std::priority_queue<Node> top_candidates;
            top_candidates=search_from_candidate_unbounded_combine(
                        Node(d_nearest, nearest), qdis, ef, &vt,fos);
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }

            int nres = 0;
            while (!top_candidates.empty()) {
                float d;
                storage_idx_t label;
                std::tie(d, label) = top_candidates.top();
                faiss::maxheap_push(++nres, D, I, d, label);
                top_candidates.pop();
            }
        }

        vt.advance();
}
}


/// standard unbounded search
std::priority_queue<HNSW::Node> HNSW::search_from_candidate_unbounded_hot_hubs_enhence(
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        idx_t n) const {
    int ndis = 0;
    int nstep = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);
    size_t cnt=0;
    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }
        nstep++;
        candidates.pop();

        size_t begin, end;
        neighbor_range(v0, 0, &begin, &end);

        // printf("begin:%ld,end:%ld\n",begin,end);
        // 遍历v0点对应的所有邻居
        size_t j = begin;
        while (j < end) {
            int v1 = neighbors[j];

            if (v1 < 0) {
                break;
            }
            // 遇到>=n 搜索末尾位置
            // 将指针指向下一组邻居结点
            if (v1 >= n)
            {
                size_t begin1,end1;
                neighbor_range(v1,0,&begin1,&end1);
                // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
                j=begin1;
                end=end1;
                continue;
            }
            j++;
            if (vt->get(v1)) {
                continue;
            }

            vt->set(v1);

            float d1 = qdis(v1);
            ++ndis;

            if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                candidates.emplace(d1, v1);
                top_candidates.emplace(d1, v1);
                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        }
    }
    return top_candidates;
}




// 全局热点，不访问热点反向边
std::priority_queue<HNSW::Node> HNSW::search_from_candidate_unbounded_hot_hubs_enhence_v2(
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        idx_t n) const {
    int ndis = 0;
    int nstep = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;
    // 已经遇到过且未访问的热点列表
    std::set<Node> hb_candidates; 
    std::unordered_set<idx_t> hbs_neighbors; // 热点邻居标记
    double base = 100000000.0;
    double base2 = 10000000.0;
    std::vector<std::string> diss;   

    // 将热点放入set
    std::unordered_set<idx_t> hbs;
    for (int i = 0; i < hot_hubs.size(); ++i)
    {
      for (auto& a:hot_hubs[i])
      {
        hbs.insert(a.first);
      }
    }
    // printf("ok!\n");

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);

    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }
        nstep++;
        candidates.pop();

        // 如果访问列该列表中的元素，就将该列
        auto it=hb_candidates.find({d0,v0});
        if (it!=hb_candidates.end())
        {
          hb_candidates.erase(it);
        }


        if (hbs.find(v0)!=hbs.end())
          diss.push_back(std::to_string(base+(double)d0));
        else if (hbs_neighbors.find(v0)!=hbs_neighbors.end())
          diss.push_back(std::to_string(base2+(double)d0)),hbs_neighbors.erase(v0);
        else
          diss.push_back(std::to_string((double)d0));


        size_t begin, end;
        neighbor_range(v0, 0, &begin, &end);

        // printf("begin:%ld,end:%ld\n",begin,end);
        // 遍历v0点对应的所有邻居
        size_t j = begin;
        while (j < end) {
            int v1 = neighbors[j];

            if (v1 < 0 || v1>=n) {
                break;
            }
            // 遇到>=n 搜索末尾位置
            // 将指针指向下一组邻居结点
            if (v1 >= n)
            {
                size_t begin1,end1;
                neighbor_range(v1,0,&begin1,&end1);
                // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
                j=begin1;
                end=end1;
                continue;
            }
            j++;
            if (vt->get(v1)) {
                continue;
            }

            if (hbs.find(v0)!=hbs.end())
              hbs_neighbors.insert(v1);

            vt->set(v1);

            float d1 = qdis(v1);
            ++ndis;

            // 如果v1 是热点，将该热点放入hb_candidates中
            if (hbs.find(v1)!=hbs.end())
            {
              hb_candidates.emplace(d1,v1);
            }

            if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                candidates.emplace(d1, v1);
                top_candidates.emplace(d1, v1);
                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        }
        if (hbs.find(v0)!=hbs.end())
          hbs.erase(v0); // 将该热点删除操作放到外部统一操作
    }
    // 热点索引，非热点搜索，标记热点邻居
    std::ofstream out("./globle_nosearch_hubs.csv",std::ofstream::app);

    for(int i=0;i<diss.size();i++){
        out<<diss[i]<<",";
    }
    out<<std::endl;
    out.close();
    return top_candidates;
}





void HNSW::search_from_candidates_hot_hubs(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        int level,
        int nres_in,unsigned n) const {
    int nres = nres_in;
    int ndis = 0;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (nres < k) {
            faiss::maxheap_push(++nres, D, I, d, v1);
        } else if (d < D[0]) {
            faiss::maxheap_pop(nres--, D, I);
            faiss::maxheap_push(++nres, D, I, d, v1);
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        size_t begin, end;
        neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = neighbors[j];
            if (v1 < 0)
                break;
            if (v1 >= n)
            {
                neighbor_range(v1,0,&begin,&end);
                continue;
            }
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);
            ndis++;
            float d = qdis(v1);
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_pop(nres--, D, I);
                faiss::maxheap_push(++nres, D, I, d, v1);
            }
            candidates.push(v1, d);
        }

        nstep++;
        if (nstep > efSearch) {
            break;
        }
    }
}



void HNSW::search_with_hot_hubs_enhence(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            VisitedTable& vt,size_t n,RandomGenerator& rng3) const{
    if (upper_beam == 1) {
        //  greedy search on upper levels 
        // 固定ep 
        storage_idx_t nearest = entry_point;
        // 随机ep（只保留0层方法）
        // storage_idx_t nearest = rng3.rand_long()%n;
        // printf("nearest : %ld\n", nearest );
        float d_nearest = qdis(nearest);

        // printf("nearest: %ld,d_nearest%f\t",nearest,d_nearest);
        for (int level = max_level; level >= 1; level--) {
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        }
        // printf("nearest: %ld,d_nearest%f\t",nearest,d_nearest);

        int ef = std::max(efSearch, 10);

        if (search_bounded_queue) {
            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);

            search_from_candidates_hot_hubs(qdis, k, I, D, candidates, 
                vt, 0, 0 ,n);
        } else {
            std::priority_queue<Node> top_candidates;
            top_candidates=search_from_candidate_unbounded_hot_hubs_enhence(
                        Node(d_nearest, nearest), qdis, ef, &vt,n);
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            // printf("top_candidates.size():%d\n",top_candidates.size());
            int nres = 0;
            while (!top_candidates.empty()) {
                float d;
                storage_idx_t label;
                std::tie(d, label) = top_candidates.top();
                faiss::maxheap_push(++nres, D, I, d, label);
                top_candidates.pop();
            }
        }

        vt.advance();
  }
}


// 统计找到最近邻时访问点个数
void HNSW::search_ndis(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            VisitedTable& vt,size_t n,RandomGenerator& rng3,std::vector<int>& ndiss,idx_t gt) const{
    if (upper_beam == 1) {
        //  greedy search on upper levels 
        // 固定ep 
        storage_idx_t nearest = entry_point;
        // 随机ep（只保留0层方法）
        // storage_idx_t nearest = rng3.rand_long()%n;
        // printf("nearest : %ld\n", nearest );
        float d_nearest = qdis(nearest);

        // printf("nearest: %ld,d_nearest%f\t",nearest,d_nearest);
        for (int level = max_level; level >= 1; level--) {
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        }
        // printf("nearest: %ld,d_nearest%f\t",nearest,d_nearest);

        int ef = std::max(efSearch, 10);

        std::priority_queue<Node> top_candidates;

        top_candidates=search_from_find_ndis(
                    Node(d_nearest, nearest), qdis, ef, &vt,ndiss,gt);
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        // printf("top_candidates.size():%d\n",top_candidates.size());
        int nres = 0;
        while (!top_candidates.empty()) {
            float d;
            storage_idx_t label;
            std::tie(d, label) = top_candidates.top();
            faiss::maxheap_push(++nres, D, I, d, label);
            top_candidates.pop();
        }

        vt.advance();
  }
}





// 热点索引不访问热点反向邻居
void HNSW::search_with_hot_hubs_enhence_v2(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            VisitedTable& vt,size_t n,RandomGenerator& rng3) const{
    if (upper_beam == 1) {
        //  greedy search on upper levels 
        // 固定ep 
        storage_idx_t nearest = entry_point;
        // 随机ep（只保留0层方法）
        // storage_idx_t nearest = rng3.rand_long()%n;
        // printf("nearest : %ld\n", nearest );
        float d_nearest = qdis(nearest);

        // printf("nearest: %ld,d_nearest%f\t",nearest,d_nearest);
        for (int level = max_level; level >= 1; level--) {
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        }
        // printf("nearest: %ld,d_nearest%f\t",nearest,d_nearest);

        int ef = std::max(efSearch, 10);

        if (search_bounded_queue) {
            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);

            search_from_candidates_hot_hubs(qdis, k, I, D, candidates, 
                vt, 0, 0 ,n);
        } else {
            std::priority_queue<Node> top_candidates;
            top_candidates=search_from_candidate_unbounded_hot_hubs_enhence_v2(
                        Node(d_nearest, nearest), qdis, ef, &vt,n);
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            // printf("top_candidates.size():%d\n",top_candidates.size());
            int nres = 0;
            while (!top_candidates.empty()) {
                float d;
                storage_idx_t label;
                std::tie(d, label) = top_candidates.top();
                faiss::maxheap_push(++nres, D, I, d, label);
                top_candidates.pop();
            }
        }

        vt.advance();
  }
}


std::priority_queue<HNSW::Node> HNSW::search_from_candidate_unbounded_hot_hubs_enhence_random(
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        idx_t n) const {
    int ndis = 0;
    int nstep = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;
    // 已经遇到过且未访问的热点列表
    std::set<Node> hb_candidates; 

    // 将热点放入set
    std::unordered_set<idx_t> hbs;
    for (int i = 0; i < hot_hubs.size(); ++i)
    {
      for (auto& a:hot_hubs[i])
      {
        hbs.insert(a.first);
      }
    }

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);
    size_t cnt=0;
    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }
        nstep++;
        cnt++;
        // 防止热点被取完
        if (cnt==10&&!hbs.empty()) // 如果在一定步长内没有碰到热点，选择已经访问的随机热点重启
        {
            if (hb_candidates.empty()) // 如果候选为空,从所有热点中随机选取一个热点重启
            {
                auto it = hbs.begin();
                v0 = *it;
                d0 = qdis(v0);
                hbs.erase(it); // 从热点列表中删去该热点，防止重复访问

            }else{  // 如果候选不为空，从已访问中选取
                auto it=hb_candidates.begin();
                v0 = (*it).second;
                d0 = (*it).first;
                hbs.erase(v0);
                hb_candidates.erase(it);
            }
            cnt=0;
        }
        else 
          candidates.pop();

        // 如果访问列该列表中的元素，就将该列
        auto it=hb_candidates.find({d0,v0});
        if (it!=hb_candidates.end())
        {
          hb_candidates.erase(it);
          //同时从热点中删除该点防止重复加入
          hbs.erase(v0);
          cnt=0;
        }


        size_t begin, end;
        neighbor_range(v0, 0, &begin, &end);

        // printf("begin:%ld,end:%ld\n",begin,end);
        // 遍历v0点对应的所有邻居
        size_t j = begin;
        while (j < end) {
            int v1 = neighbors[j];

            if (v1 < 0) {
                break;
            }
            // 遇到>=n 搜索末尾位置
            // 将指针指向下一组邻居结点
            if (v1 >= n)
            {
                size_t begin1,end1;
                neighbor_range(v1,0,&begin1,&end1);
                // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
                j=begin1;
                end=end1;
                continue;
            }
            j++;
            if (vt->get(v1)) {
                continue;
            }

            vt->set(v1);

            float d1 = qdis(v1);
            ++ndis;

            // 如果v1 是热点，将该热点放入hb_candidates中
            if (hbs.find(v1)!=hbs.end())
            {
              hb_candidates.emplace(d1,v1);
            }

            if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                candidates.emplace(d1, v1);
                top_candidates.emplace(d1, v1);
                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        }
    }
    return top_candidates;
}


// 随机热点 统计热点信息
std::priority_queue<HNSW::Node> HNSW::search_from_candidate_unbounded_hot_hubs_enhence_random_v2(
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        idx_t n) const {
    int ndis = 0;
    int nstep = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;
    // 已经遇到过且未访问的热点列表
    std::set<Node> hb_candidates; 
    std::unordered_set<idx_t> hbs_neighbors; // 热点邻居标记
    double base = 100000000.0;
    double base2 = 10000000.0;
    std::vector<std::string> diss;   

    // 将热点放入set
    std::unordered_set<idx_t> hbs;
    for (int i = 0; i < hot_hubs.size(); ++i)
    {
      for (auto& a:hot_hubs[i])
      {
        hbs.insert(a.first);
      }
    }

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);
    size_t cnt=0;
    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }
        nstep++;
        cnt++;
        // 防止热点被取完
        if (cnt%10==0&&!hbs.empty()) // 如果在一定步长内没有碰到热点，选择已经访问的随机热点重启
        {
            if (hb_candidates.empty()) // 如果候选为空,从所有热点中随机选取一个热点重启
            {
                auto it = hbs.begin();
                v0 = *it;
                d0 = qdis(v0);
                // hbs.erase(it); // 从热点列表中删去该热点，防止重复访问

            }else{  // 如果候选不为空，从已访问中选取
                auto it=hb_candidates.begin();
                v0 = (*it).second;
                d0 = (*it).first;
                // hbs.erase(v0);
                hb_candidates.erase(it);
            }
            cnt=0;
        }
        else 
          candidates.pop();

        // 如果访问列该列表中的元素，就将该列
        auto it=hb_candidates.find({d0,v0});
        if (it!=hb_candidates.end())
        {
          hb_candidates.erase(it);
          //同时从热点中删除该点防止重复加入
          // hbs.erase(v0);
          cnt=0;
        }

        if (hbs.find(v0)!=hbs.end())
          diss.push_back(std::to_string(base+(double)d0));
        else if (hbs_neighbors.find(v0)!=hbs_neighbors.end())
          diss.push_back(std::to_string(base2+(double)d0)),hbs_neighbors.erase(v0);
        else
          diss.push_back(std::to_string((double)d0));


        size_t begin, end;
        neighbor_range(v0, 0, &begin, &end);

        // printf("begin:%ld,end:%ld\n",begin,end);
        // 遍历v0点对应的所有邻居
        size_t j = begin;
        while (j < end) {
            int v1 = neighbors[j];

            if (v1 < 0) {
                break;
            }
            // 遇到>=n 搜索末尾位置
            // 将指针指向下一组邻居结点
            if (v1 >= n)
            {
                size_t begin1,end1;
                neighbor_range(v1,0,&begin1,&end1);
                // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
                j=begin1;
                end=end1;
                continue;
            }
            j++;
            if (vt->get(v1)) {
                continue;
            }

            if (hbs.find(v0)!=hbs.end())
              hbs_neighbors.insert(v1);

            vt->set(v1);

            float d1 = qdis(v1);
            ++ndis;

            // 如果v1 是热点，将该热点放入hb_candidates中
            if (hbs.find(v1)!=hbs.end())
            {
              hb_candidates.emplace(d1,v1);
            }

            if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                candidates.emplace(d1, v1);
                top_candidates.emplace(d1, v1);
                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        }
        if (hbs.find(v0)!=hbs.end())
          hbs.erase(v0); // 将该热点删除操作放到外部统一操作
    }
    // 将距离保存到文件
    std::ofstream out("./hnsw_outdatas.csv",std::ofstream::app);

    for(int i=0;i<diss.size();i++){
      out<<diss[i]<<",";
    }
    out<<std::endl;
    out.close();
    return top_candidates;
}


// 全局热点 统计热点信息
std::priority_queue<HNSW::Node> HNSW::search_from_candidate_unbounded_hot_hubs_enhence_random_v3(
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        idx_t n,std::unordered_map<idx_t,int>& fqc) const{
    int ndis = 0;
    int nstep = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;
    // 已经遇到过且未访问的热点列表
    std::set<Node> hb_candidates; 
    std::unordered_set<idx_t> hbs_neighbors; // 热点邻居标记
    double base = 100000000.0;
    double base2 = 10000000.0;
    std::vector<std::string> diss;   

    // 将热点放入set
    std::unordered_set<idx_t> hbs;
    for (int i = 0; i < hot_hubs.size(); ++i)
    {
      for (auto& a:hot_hubs[i])
      {
        hbs.insert(a.first);
      }
    }
    // printf("ok!\n");

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);

    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }
        nstep++;
        candidates.pop();

        // 如果访问列该列表中的元素，就将该列
        auto it=hb_candidates.find({d0,v0});
        if (it!=hb_candidates.end())
        {
          hb_candidates.erase(it);
        }


        if (hbs.find(v0)!=hbs.end()){
          diss.push_back(std::to_string(base+(double)d0));
          fqc[v0]++;
        }
        else if (hbs_neighbors.find(v0)!=hbs_neighbors.end())
          diss.push_back(std::to_string(base2+(double)d0)),hbs_neighbors.erase(v0);
        else
          diss.push_back(std::to_string((double)d0));


        size_t begin, end;
        neighbor_range(v0, 0, &begin, &end);

        // printf("begin:%ld,end:%ld\n",begin,end);
        // 遍历v0点对应的所有邻居
        size_t j = begin;
        while (j < end) {
            int v1 = neighbors[j];

            if (v1 < 0) {
                break;
            }
            // 遇到>=n 搜索末尾位置
            // 将指针指向下一组邻居结点
            if (v1 >= n)
            {
                size_t begin1,end1;
                neighbor_range(v1,0,&begin1,&end1);
                // printf("v1:%d,begin:%ld,end:%ld,begin1:%ld,end1:%ld\n",v1,begin,end,begin1,end1);
                j=begin1;
                end=end1;
                continue;
            }
            j++;
            if (vt->get(v1)) {
                continue;
            }

            if (hbs.find(v0)!=hbs.end())
              hbs_neighbors.insert(v1);

            vt->set(v1);

            float d1 = qdis(v1);
            ++ndis;

            // 如果v1 是热点，将该热点放入hb_candidates中
            if (hbs.find(v1)!=hbs.end())
            {
              hb_candidates.emplace(d1,v1);
            }

            if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                candidates.emplace(d1, v1);
                top_candidates.emplace(d1, v1);
                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        }
        if (hbs.find(v0)!=hbs.end())
          hbs.erase(v0); // 将该热点删除操作放到外部统一操作
    }
    // 将距离保存到文件
    std::ofstream out("./globle_hubs.csv",std::ofstream::app);

    for(int i=0;i<diss.size();i++){
        out<<diss[i]<<",";
    }
    out<<std::endl;
    out.close();
    return top_candidates;
}





void HNSW::search_with_hot_hubs_enhence_random(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            VisitedTable& vt,size_t n,RandomGenerator& rng3,std::unordered_map<idx_t,int>& fqc) const{
    if (upper_beam == 1) {
        //  greedy search on upper levels 
        // 固定ep 
        storage_idx_t nearest = entry_point;
        // 随机ep（只保留0层方法）
        // storage_idx_t nearest = rng3.rand_long()%n;
        // printf("nearest : %ld\n", nearest );
        float d_nearest = qdis(nearest);

        // printf("nearest: %ld,d_nearest%f\t",nearest,d_nearest);
        for (int level = max_level; level >= 1; level--) {
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        }
        // printf("nearest: %ld,d_nearest%f\t",nearest,d_nearest);

        int ef = std::max(efSearch, 10);

        if (search_bounded_queue) {
            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);

            search_from_candidates_hot_hubs(qdis, k, I, D, candidates, 
                vt, 0, 0 ,n);
        } else {
            std::priority_queue<Node> top_candidates;
            // printf("%d\n",hot_hubs[0].size());
            // v3: 统计距离变化：全局热点搜索方法v_3
            // v2: 统计距离变化：局部随机热点搜索方法v_2
            top_candidates=search_from_candidate_unbounded_hot_hubs_enhence_random_v3(
                        Node(d_nearest, nearest), qdis, ef, &vt,n,fqc);
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            // printf("top_candidates.size():%d\n",top_candidates.size());
            int nres = 0;
            while (!top_candidates.empty()) {
                float d;
                storage_idx_t label;
                std::tie(d, label) = top_candidates.top();
                faiss::maxheap_push(++nres, D, I, d, label);
                top_candidates.pop();
            }
        }

        vt.advance();
}
}





// index 相似度比较

/*void HNSW::similarity(HNSW * index1,HNSW * index2,int nb){
  float res=0;
  float index1_sum=0,index2_sum=0;

  for (int i = 0; i <nb ; ++i)
  {
      float cnt=0;
      size_t begin1, end1;
      index1->neighbor_range(i, 0, &begin1, &end1);
      size_t begin2, end2;
      index2->neighbor_range(i, 0, &begin2, &end2);  
      //std::cout<<end1-begin1<<"：："<<end2-begin2<<std::endl;

      // index1中，访问点个数
      for (size_t x = begin1; x < end1; ++x) {
          if (index1->neighbors[x] < 0) {
            break;
          }
          index1_sum++;
      }

      // index2中，访问点个数
      for (size_t x = begin2; x < end2; ++x) {

          if (index2->neighbors[x] < 0) {
            break;
          }
          index2_sum++;
      }



      for (size_t x = begin1; x < end1; ++x) {
          if (index1->neighbors[x] < 0) {
            break;
          }
        for (size_t y = begin2; y < end2; ++y){
            if (index2->neighbors[y] < 0) {
              break;
            }
            if(index1->neighbors[x] == index2->neighbors[y]){
              cnt++;

              
              continue;
            }
                
        }
      
      }

      res+=cnt;
      
}
  std::cout<<res<<std::endl;

  std::cout<<index1_sum<<std::endl;

  std::cout<<index2_sum<<std::endl;

  std::cout<<res*2/(index1_sum+index2_sum)<<std::endl;

}*/


// 统计索引中每个点出现频率
void HNSW::similarity(HNSW * index1,HNSW * index2,int nb){
  
  std::map<size_t,size_t> ma;
  long long sum=0;
  for (int i = 0; i < nb ; ++i)
  {
      size_t begin1, end1;
      index1->neighbor_range(i, 0, &begin1, &end1);
      int cnt=5;
      for (size_t x = begin1; x < end1&&cnt; ++x) {
          if (index1->neighbors[x] < 0) {
              break;
            }
          cnt--;
          ma[index1->neighbors[x]]++;
          sum++;
      }
  }
  for (auto a:ma)
  {
    std::cout<<"sum: "<<sum<<"  id: "<<a.first<<" cnt: "<<a.second<<std::endl;
  }
}


void HNSW::setIndex(HNSW* findex,HNSW* sindex){
      index1 = findex;
      index2 = sindex;
}


void HNSW::MinimaxHeap::push(storage_idx_t i, float v) {
    // 如果放入的节点数大于最大数（n）
    if (k == n) {
        if (v >= dis[0])
            return;
        faiss::heap_pop<HC>(k--, dis.data(), ids.data());
        --nvalid;
    }
    // 不足n个直接插入
    faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
    ++nvalid;
}

// dis[0] 存放最大距离
float HNSW::MinimaxHeap::max() const {
    return dis[0];
}

int HNSW::MinimaxHeap::size() const {
    return nvalid;
}

void HNSW::MinimaxHeap::clear() {
    nvalid = k = 0;
}

// ids，dis为MinimaxHeap的变量，用来存储candidates中的值
int HNSW::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);  
    // returns min. This is an O(n) operation
    int i = k - 1;
    // 寻找ids中第一个不为-1的值（ids为结果存放数组）
    while (i >= 0) {
        if (ids[i] != -1)
            break;
        i--;
    }
    // 如果全为-1，意味着没有元素
    if (i == -1)
        return -1;

    // 存放最小值下标
    int imin = i;
    // 存放最小值
    float vmin = dis[i];
    i--;
    // 寻找最小值
    while (i >= 0) {
        if (ids[i] != -1 && dis[i] < vmin) {
            vmin = dis[i];
            imin = i;
        }
        i--;
    }
    // 用vmin_out 记录最小值的值
    if (vmin_out)
        *vmin_out = vmin;

    // ret为最小值的id
    int ret = ids[imin];

    // 将此最小值位置置为-1，相当于弹出
    ids[imin] = -1;
    --nvalid;

    return ret;
}

// 寻找当前距离中，小于阈值thresh中值的个数的点
int HNSW::MinimaxHeap::count_below(float thresh) {
// O(N)
    int n_below = 0;
    for (int i = 0; i < k; i++) {
        if (dis[i] < thresh) {
            n_below++;
        }
    }

    return n_below;
}

} // namespace faiss


