# 一、哈希

## [1. 两数之和 - 力扣（LeetCode）]([1. 两数之和 - 力扣（LeetCode）](https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked))

### 题解

> **法一（最优）：**哈希表。
>
> - 因为数据保证有且仅有一组解，假设解为 $[i, j], (i < j)$ ，则当我们遍历到 $j$ 时， $nums[i]$ 一定在哈希表中。
>
> **法二：**双指针。

### CODE

```c++
class Solution { // 法一：哈希表
public:
    vector<int> twoSum(vector<int>& nums, int target)
    {
        vector<int> res;
        unordered_map<int,int> hash;
        for (int i = 0; i < nums.size(); i ++ )
        {
            int another = target - nums[i];
            if (hash.count(another))
            {
               	res.push_back(hash[another]);
                res.push_back(i);
                break;
            }
            hash[nums[i]] = i;
        }
        return res;
    }
};

class Solution { // 法二：双指针
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<pair<int, int>> p;
        vector<int> res;

        for (int i = 0; i < nums.size(); i ++) p.push_back({nums[i], i});
        sort(p.begin(), p.end());

        int i = 0, j = p.size() - 1;
        while (i < p.size() && j >= 0) {
            if(p[i].first + p[j].first < target) i ++;
            else if ((p[i].first + p[j].first > target)) j --;
            else {
                res.push_back(min(p[i].second, p[j].second));
                res.push_back(max(p[i].second, p[j].second));
                break;
            }
            
        }
        return res;
    }
};
```



## [49. 字母异位词分组 - 力扣（LeetCode）](https://leetcode.cn/problems/group-anagrams/description/?envType=study-plan-v2&envId=top-100-liked)

### 题解

> **法一：**哈希表。
>
> - 由相同字母组成的字符串 $str$ ，排序后字符串 $key$ 都相等，将 $key$ 作为键， $str$ 作为值，维护一个 `unordered_map<string, vector<string>>` 的哈希表，每个 $key$ 对应，由相同字母组成的字符串 $str$ ，所组成的列表。

### CODE

```c++
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> dict;
        for (auto it: strs) {
            string key = it;
            sort(key.begin(), key.end());
            dict[key].push_back(it);
        }

        vector<vector<string>> res;
        for(auto it: dict) {
            res.push_back(it.second);
        }
        return res;
    }
};
```



## [128. 最长连续序列 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-consecutive-sequence/description/?envType=study-plan-v2&envId=top-100-liked)

### 题解

> **法一：**集合。
>
> - 首先将所有数字放入集合，遍历集合中的元素，因为要找连续的数字序列，因此可以通过向后枚举相邻的数字（即不断加一），判断后面一个数字是否在集合中即可；
>
> - 为了保证 $O(n)$ 的复杂度，需避免重复枚举序列，因此只对序列的起始数字向后枚举（例如 $[1,2,3,4]$ ，只对 $1$ 枚举， $2$ ， $3$ ， $4$ 时跳过），因此需要判断一下是否是序列的起始数字（即判断一下 $num-1$ 是否在集合中，若不在，则为起始数字）。

### CODE

```c++
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> nums_set(nums.begin(), nums.end());

        int res = 0;
        for (auto num: nums_set) {
            if (!nums_set.count(num - 1)) { // 只考虑连续序列的起始元素，避免重复遍历序列
                int end = num;
                while (nums_set.count(end)) end ++;
                res = max(res, end - num);
            }
        }
        return res;
    }
};
```



# 二、双指针

## [283. 移动零 - 力扣（LeetCode）](https://leetcode.cn/problems/move-zeroes/description/?envType=study-plan-v2&envId=top-100-liked)

### 题解

> **法一：**双指针。
>
> - 指针 $r$ 表示当前访问到的位置，指针 $l$ 表示当前第一个可以放置**非零**元素的位置。

### CODE

```c++
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int l = 0;
        for (int r = 0; r < nums.size(); r ++) {
            if (nums[r]) nums[l ++] = nums[r];
        }

        while (l < nums.size()) nums[l ++] = 0;
    }
};
```



## [11. 盛最多水的容器 - 力扣（LeetCode）](https://leetcode.cn/problems/container-with-most-water/description/?envType=study-plan-v2&envId=top-100-liked)

### 题解

>**法一：**双指针。
>
>- 最开始的时候，如果我们用指针 $l$ 和 $r$ 指向最两端的直线，此时两条直线之间的距离就是最大的，即我们所求矩形面积的宽度 $width$ 为最大；
>- 但是位于最两端的直线不一定是最高的，所以它们组成矩形的面积也就不一定是最大的。因此我们依然需要继续遍历整个数组，这时我们将指向数组两端的指针慢慢往里面收敛，直到找到面积最大值；
>- 对于此时 $l$ 和 $r$ 指向的直线，他们之间的宽度已经是最宽了。于是在收敛的过程中，如果遇到的高度比两端的柱子更低的话，由于之间的宽度更短，所以面积必定更小，我们就可以直接跳过，不予考虑。我们只需要考虑收敛时出现的那些高度更高的柱子；
>- 该方法在双指针向中间收敛的过程中，对数组中的每个元素只访问了一次，因此时间复杂度为 $O(n)$ 。

### CODE

```c++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int l = 0, r = height.size() - 1;
        int res = 0;
        while (l < r)
        {
            int h = min(height[l], height[r]);
            res = max(res, h * (r - l));
            while (l < r && height[l] <= h) l ++;
            while (l < r && height[r] <= h) r --;
        }
        return res;
    }
};
```



## [15. 三数之和 - 力扣（LeetCode）](https://leetcode.cn/problems/3sum/description/?envType=study-plan-v2&envId=top-100-liked)

### 题解

>**法一：**排序+双指针。
>
>- 枚举每个数，表示该数 $nums[i]$ 已被确定，在排序后的情况下，通过双指针 $l$ ， $r$ 分别从左边 $l = i + 1$ 和右边 $n - 1$ 往中间靠拢，找到 $nums[i] + nums[l] + nums[r] == 0$ 的所有符合条件的搭配。
>- 在找符合条件搭配的过程中，假设 $sum = nums[i] + nums[l] + nums[r]$ ，
>   - 若 $sum > 0$ ，则 $r$ 往左走，使 $sum$ 变小；
>   - 若 $sum < 0$ ，则$l$往右走，使 $sum$ 变大；
>   - 若 $sum == 0$ ，则表示找到了与 $nums[i]$ 搭配的组合 $nums[l]$ 和 $nums[r]$ ，存到 $res$ 中；
>- 判重处理：
>   - 当 $nums[i] == nums[i - 1]$ ，表示当前确定好的数与上一个一样，需要直接 $continue$ ；
>   - 当找符合 $sum == 0$ 时，需要对 $nums[l]$ 和 $nums[r]$ 进行判重。

### CODE

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> all_res;
        for (int i = 0; i < nums.size(); i ++) {
            if (i != 0 && nums[i] == nums[i - 1]) continue; // 判重
            
            int l = i + 1, r = nums.size() - 1;
            while (l < r) {
                int sum = nums[i] + nums[l] + nums[r];
                
                if (sum > 0) r --;
                else if (sum < 0) l ++;
                else {
                    vector<int> res = {nums[i], nums[l], nums[r]};
                    all_res.push_back(res);
                    // 对 nums[l] 和 nums[r] 进行判重
                    while (l < r && nums[++ l] == nums[l - 1]);
                    while (l < r && nums[-- r] == nums[r + 1]);
                }
            }
        }
        return all_res;
    }
};
```



## [42. 接雨水 - 力扣（LeetCode）](https://leetcode.cn/problems/trapping-rain-water/description/?envType=study-plan-v2&envId=top-100-liked)

### 题解

>**法一：**三次线性扫描。
>
>- 观察整个图形，考虑对水的面积按**列**进行拆解；
>
>- 注意到，每个矩形条上方所能接受的水的高度，是由它**左边最高**的矩形，和**右边最高**的矩形决定的。具体地，假设第 $i$ 个矩形条的高度为 $height[i]$ ，且矩形条左边最高的矩形条的高度为 $left_{max}[i]$ ，右边最高的矩形条高度为 $right_{max}[i]$ ，则该矩形条上方能接受水的高度为 $min(left_{max}[i],right_{max}[i]) - height[i]$ ；
>
>- 需要分别从左向右扫描求 $left_{max}$ ，从右向左求 $right_{max}$ ，最后统计答案即可；
>
>- 注意特判 $n$ 为 $0$ 。

### CODE

```c++
class Solution {
public:
    int trap(vector<int>& height) {
        int n = height.size(), ans = 0;
        if (n == 0)
            return 0;

        vector<int> left_max(n), right_max(n);

        left_max[0] = height[0];
        for (int i = 1; i < n; i++)
            left_max[i] = max(left_max[i - 1], height[i]);

        right_max[n - 1] = height[n - 1];
        for (int i = n - 2; i >= 0; i--)
            right_max[i] = max(right_max[i + 1], height[i]);

        for (int i = 0; i < n; i++)
            ans += min(left_max[i], right_max[i]) - height[i];

        return ans;
    }
};
```



# 三、滑动窗口

## [3. 无重复字符的最长子串 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/?envType=study-plan-v2&envId=top-100-liked)

### 题解

> **法一：**双指针。
>
> - 定义指针 $l$ 和指针 $r$ ，表示当前扫描到的子串是 $[l, r]$ （闭区间）。扫描过程中维护一个哈希表，表示 $[i,j] $中每个字符出现的次数。
> - 线性扫描时，每次循环的流程如下：
>   - 指针 $r$ 向后移一位，同时将哈希表中 $s[r]$ 的计数加一： $hash[s[r]]++$ ；
>   - 假设 $j$ 移动前的区间 $[l, r]$ 中没有重复字符，则 $r$ 移动后，只有 $s[r]$ 可能出现 $2$ 次。因此我们不断向后移动 $l$ ，直至区
>     间 $[l, r]$ 中  $s[r]$ 的个数等于 $1$ 为止。
> - 复杂度分析：由于 $l$ ,  $r$ 均最多增加 $n$ 次，且哈希表的插入和更新操作的复杂度都是 $O(1)$ ，因此，总时间复杂度 $O(n)$ 。

### CODE

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char, int> hash;
        int res = 0;
        for (int l = 0, r = 0; r < s.size(); r ++) {
            hash[s[r]] ++;
            while (hash[s[r]] > 1) hash[s[l ++]] --;
            res = max(res, r - l + 1);
        }
        return res;
    }
};
```



## [438. 找到字符串中所有字母异位词 - 力扣（LeetCode）](https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/?envType=study-plan-v2&envId=top-100-liked)

### 题解

> **法一：**双指针。
>
> - 使用哈希表记录 $p$ 的每个字符个数；
> - 使用指针 $l$ 和指针 $r$ 维护一个固定长度的滑动窗口，使用 $cnt$ 去维护窗口中有多少字符可以作为 $p$ 的异位字符，当 $cnt==p.size()$ 时，当前窗口构成的子串是 $p$ 的异位词：
>   - 指针 $r$ 向右访问 $s[r]$ 时，记录窗口中字符 $s[r]$ 的数量，若小于等于 $p$ 中该字符的数量，则说明 $s[r]$ 可以作为 $p$ 的一个异位字符，则 $cnt++$ ；
>   - 维护固定长度的滑动窗口：
>     - 若 $s[l]$ 的字符数量，小于等于 $p$ 中该字符的数量，说明 $s[l]$ 是 $p$ 的一个异位字符，则 $cnt--$ ；
>     - 指针 $l$ 右移，并维护窗口中字符 $s[l]$ 的数量，即 $sHash[s[l ++]] --$ 。

### CODE

```c++
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        unordered_map<char, int> sHash, pHash;
		// 记录 p 的每个字符个数
        for (auto it: p) pHash[it] ++;

        vector<int> res;
        int cnt = 0;
        for (int l = 0, r = 0; r < s.size(); r ++) {
            sHash[s[r]] ++;
            
            // s[r] 可以作为 p 的一个异位字符，cnt ++
            if (sHash[s[r]] <= pHash[s[r]]) cnt ++;
            
            //维护固定长度的滑动窗口
            if (r - l + 1 > p.size()) {
                if (sHash[s[l]] <= pHash[s[l]]) cnt --;
                sHash[s[l ++]] --;
            }
            
            if (cnt == p.size()) res.push_back(l);
        }
        return res;
    }
};
```



# 四、子串

## [560. 和为 K 的子数组 - 力扣（LeetCode）](https://leetcode.cn/problems/subarray-sum-equals-k/description/?envType=study-plan-v2&envId=top-100-liked)

### 题解

> **法一：**前缀和+哈希表。
>
> - 对原数组求前缀和后，一个和为 $k$ 的子数组即为**一对前缀和的差值为 $k$ 的位置**。
> - 遍历前缀和数组，用哈希表记录每个前缀和出现的次数。特别地，初始时前缀和为 $0$ 需要被额外记录一次。
> - 遍历过程中，对于当前前缀和 $tot$，累加之前 $tot - k$ 前缀和出现的次数（ $tot-(tot-k)==k$ ）。

### CODE

```c++
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> hash;
        hash[0] = 1;
        
        int tot = 0, res = 0;
        for (auto it: nums) {
            tot += it; // 计算前缀和
            res += hash[tot - k]; // 若存在和为 tot - k 的前缀和，则存在一个子数组和为 k 
            hash[tot] ++;
        }
        
        return res;
    }
};
```



## [239. 滑动窗口最大值 - 力扣（LeetCode）](https://leetcode.cn/problems/sliding-window-maximum/description/?envType=study-plan-v2&envId=top-100-liked)

### 题解

> **法一：**单调队列模板题。

### CODE

```c++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();

        vector<int> res, q(n + 7, 0); // 队列 q 存储的是下标
        int hh = 0, tt = -1;

        for (int i = 0; i < n; i ++) {
            if (hh <= tt && i - q[hh] + 1 > k) hh ++;
            while (hh <= tt && nums[q[tt]] <= nums[i]) tt --; // 注意，访问数组元素时，切记使用 nums[] 嵌套 q[] ，因为 q[] 存储的是下标
            q[++ tt] = i;
            if (i >= k - 1) res.push_back(nums[q[hh]]);
        }
        return res;
    }
};
```

