## [合并两个有序数组](https://leetcode.cn/problems/merge-sorted-array/description/?envType=study-plan-v2&envId=top-interview-150)

### 题解

> **逆向双指针**：将指针设置为从后向前遍历，每次取两者之中的较大者放进 `nums1` 的尾部，即可避免使用辅助空间。

### CODE

```c++
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int i = m - 1, j = n - 1, k = m + n - 1; //尾部指针
        
        while (i >= 0 && j >= 0) 
            if (nums1[i] >= nums2[j]) nums1[k --] = nums1[i --];
            else nums1[k --] = nums2[j --];
    
        //若nums1中有元素未访问，无需操作，因为最终结果本身便存储在nums1中
        while (j >= 0) nums1[k --] = nums2[j --]; 
    }
};
```



## [移除元素](https://leetcode.cn/problems/remove-element/description/?envType=study-plan-v2&envId=top-interview-150)

### 题解

> **法一（最坏情况左右指针各遍历一次数组）**：左指针`left`、右指针`right`初始值均为$0$，`left`表示元素的插入位置，`right`向后遍历，若`nums[right] != val`，即为`nums[right]`为需要的元素，插入`left`处，`left ++`。
>
> **法二（优化，最坏情况下仅遍历一次数组）**：左指针`left`初始为$0$，右指针`right`初始为数组尾部，`left`向后遍历，若`nums[left] == val`，即`nums[left]`为不需要的元素，用`nums[right]`置换该值，若`nums[right] == val`，则`left`处仍为不需要元素，需再次替换，故`left --`，当`left > right`时，算法结束。

### CODE

```c++
class Solution { //法一：判断元素值与目标值不相等
public:
    int removeElement(vector<int>& nums, int val) {
        int left = 0;
        for (int right = 0; right < nums.size(); right ++) {
            if (nums[right] != val) {
                nums[left ++] = nums[right];
            }
        }
        return left;
    }
};

class Solution { //法二：判断元素值与目标值相等
public:
    int removeElement(vector<int>& nums, int val) {
        int left, right = nums.size() - 1;
        for (left = 0; left <= right; left ++) {
            if (nums[left] == val) {
                nums[left] = nums[right --];
            }
            if (nums[left] == val) left --;

        }
        return left;
    }
};
```



## [删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/description/?envType=study-plan-v2&envId=top-interview-150)

### 题解

>`k`指向当前插入位置，遍历数组，若元素`elem`不等于`nums[k - 1]`，则`elem`为第一个首次出现的元素，插入`k`处。当`k == 0`时，前面无元素，不可能出现重复元素，直接插入。

### CODE

```c++
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int k = 0;
        for (auto elem : nums) {
            if (k < 1 || nums[k - 1] != elem) 
                nums[k ++] = elem;
        }
        return k;
    }
};
```



## [删除有序数组中的重复项Ⅱ](https://leetcode.cn/problems/remove-duplicates-from-sorted-array-ii/description/?envType=study-plan-v2&envId=top-interview-150)

### 题解

> 与上题类似，当`k == 1`时，前面仅一个元素，不可能出现两个重复值，故`k < 2`时，直接插入。

### CODE

```c++
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int k = 0;
        for (auto elem : nums) {
            if (k < 2 || nums[k - 2] != elem) 
                nums[k ++] = elem;
        }
        return k;
    }
};
```



## [多数元素](https://leetcode.cn/problems/majority-element/description/?envType=study-plan-v2&envId=top-interview-150)

### 题解

> **法一**：排序。
>
> **法二**：哈希。
>
> **法三（时空最优解）**：寻找数组中超过一半的数字，即该数字出现的次数比数组中其它数字出现次数的总和都大。也就是说，如果将数组中要查找的众数替换成$1$，其它数字替换成$-1$，若让其相加，最后的值肯定大于$0$。
> 因此，可以执行以下操作：
>
> - 设置两个变量`res`和`cnt`，`res`用来保存数组中遍历到的某个数字，`cnt`表示`res`当前出现的次数。
>
> - 遍历整个数组：
>
>   - 如果数字与$res$保存的数字相同，则`cnt ++`；
>
>   - 如果数字与$res$保存的数字不同，则`cnt --`；
>
>   - 如果出现次数`cnt == 0`，则`res`变为当前遍历的元素，同时令`cnt == 1`。
>
> - 遍历完数组中的所有元素即可返回`res`的值。

### CODE

```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int res, cnt = 0;
        for (auto elem : nums) {
            if (!cnt) res = elem, cnt = 1;
            else if (res == elem) cnt ++;
            else cnt --;
        }
        return res;
    }
};
```



## [轮转数组](https://leetcode.cn/problems/rotate-array/description/?envType=study-plan-v2&envId=top-interview-150)

### 题解

> 注意`k %= nums.size()`即可。

### CODE

```c++
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        k %= nums.size();
        reverse(nums.begin(), nums.end());
        reverse(nums.begin(), nums.begin() + k);
        reverse(nums.begin() + k, nums.end());
    }
};
```



## [买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/?envType=study-plan-v2&envId=top-interview-150)

### 题解

> 枚举每一天卖出的情况，当日卖出的最大收益为当前`price`减前面股票的最小值`minp`，并取最大值为`res`。

### CODE

```
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int res = 0, minp = prices[0];
        for (auto price : prices) { 
            res = max(res, price - minp);
            minp = min(minp, price);
        }
        return res;
    }
};
```



## [买卖股票的最佳时机Ⅱ](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/description/?envType=study-plan-v2&envId=top-interview-150)

### 题解

> 分解每次交易：假设在**第二天购买股票，第四天出售股票**（$p_4-p_2$），等价于，**第二天购买股票，第三天出售该股票**（$p_3-p_2$），然后**再次购买，第四天再出售**（$p_4-p_2=p_4-p_3+p_3-p_2$），即每次交易可分解为购买股票后隔天出售，然后再次购买。则，只需枚举隔天出售股票的收益即可，若为正收益，则`res += prices[i] - prices[i - 1]`。

### CODE

```
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int res = 0;
        for (int i = 1; i < prices.size(); i ++) {
            res += max(0, prices[i] - prices[i - 1]);
        }
        return res;
    }
};
```



## [跳跃游戏](https://leetcode.cn/problems/jump-game/description/?envType=study-plan-v2&envId=top-interview-150)

### 题解

> 枚举跳到了第`i`个下标，`far`表示通过前面元素可到达的最远下标。

### CODE

```
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int far = nums[0];
        for (int i = 1; i < nums.size(); i ++) {
            if (i > far) return false; 
            far = max(far, i + nums[i]);
        }
        return true;
    }
};
```



## [跳跃游戏Ⅱ](https://leetcode.cn/problems/jump-game-ii/description/?envType=study-plan-v2&envId=top-interview-150)

### 题解

> **法一**：`f[i]`表示从下标$0$到达下标`i`所需要的最小跳数。当位于下标`i`时，通过`i`可以跳跃至`j`，状态转移方程：`f[j] = min(f[j], f[i] + 1)`。从下标$0$​开始跳跃，故初始状态`f[0] = 0`。
>
> **法二（优化）**：`far`表示通过前面元素能到达的最远下标，`last_far`表示通过上一次跳跃可达到的最远端。枚举`i`~`last_far`的元素，

### CODE

```c++
class Solution { //法一
public:
    int jump(vector<int>& nums) {
        int n = nums.size();
        vector<int> f(n, INT_MAX);
        f[0] = 0;

        for (int i = 0; i < n; i ++) {
            for (int j = i + 1; j <= i + nums[i]; j ++) 
                if (j < n) f[j] = min(f[j], f[i] + 1);
        }

        return f[n - 1];
    }
};

class Solution { //法二
public:
    int jump(vector<int>& nums) {
        int far = 0, last_far = 0, res = 0;
        
        for (int i = 0; i < nums.size() - 1; i ++)
        {
            far = max(far, i + nums[i]);
            if (i == last_far)
            {
                last_far = far;  
                res ++;         
            }
        }
        return res;
    }
};
```



