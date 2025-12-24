# 一、背包问题

## [AcWing 2. 01背包问题 - AcWing](https://www.acwing.com/activity/content/problem/content/997/)

### 题解

> **法一：**$DP$。
>
> - 状态 $dp[i][j]$ 表示仅考虑前 $i$ 个物品，背包容量 $j$ 下的最大价值。
> - 状态转移：
>   - **选**第 $i$ 个物品：$dp[i][j] = dp[i - 1][j - v[i]] + w[i]$。
>   - **不选**第 $i$ 个物品：$dp[i][j] = dp[i - 1][j]]$。

### CODE

```c++
#include <bits/stdc++.h>

using namespace std;

const int N = 1e3 + 7;

int n, m;
int v[N], w[N];
int dp[N][N];

int main() {
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++) scanf("%d%d", &v[i], &w[i]);

    for (int i = 1; i <= n; i ++) {
        for (int j = 1; j <= m; j ++) {
            if (j >= v[i]) dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - v[i]] + w[i]);
            else dp[i][j] = dp[i - 1][j]; // 容量不够选第 i 个物品。
        }
    }
    
    printf("%d", dp[n][m]);
    return 0;
}

/*
优化v[N]、w[N]的存储空间：因为每次仅访问 v[i]，w[i]，故可用 v，w 代替数组。
*/
int n, m, v, w;
int dp[N][N];

int main() {
    scanf("%d%d", &n, &m);

    for (int i = 1; i <= n; i ++) {
        scanf("%d%d", &v, &w);
        
        for (int j = 1; j <= m; j ++) {
            if (j >= v) dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - v] + w);
            else dp[i][j] = dp[i - 1][j]; // 容量不够选第 i 个物品。
        }
    }
    
    printf("%d", dp[n][m]);
    return 0;
}

/*
优化 dp[N][N] 的存储空间：每次仅使用dp[i - 1]，故可优化为滚动数组。
*/
int n, m, v, w;
int dp[N];

int main() {
    scanf("%d%d", &n, &m);

    for (int i = 1; i <= n; i ++) {
        scanf("%d%d", &v, &w);
        
        // 每次只访问 dp[j - v]，且每个物品最多取一次，故按容量降序更新 dp[j]。
        for (int j = m; j >= v; j --) dp[j] = max(dp[j], dp[j - v] + w);
    }
    
    printf("%d", dp[m]);
    return 0;
}
```



## [3. 完全背包问题 - AcWing题库](https://www.acwing.com/problem/content/3/)

### 题解

> **法一：**$DP$。
>
> - $dp[i][j] = max(dp[i-1][j],~dp[i-1][j-v]+w,~dp[i-1][j-2*v]+2*w,~dp[i-1][j-3*v]+3*w,~.....)$ 。
> - $dp[i][j-v]= max(dp[i-1][j-v],~dp[i-1][j-2*v] + w,~dp[i-1][j-3*v]+2*w,~.....)$。
> - 易得状态转移方程 $dp[i][j] = max(dp[i][j-v]+w,dp[i-1][j])$。
> - 空间优化同 $01$ 背包。

### CODE

```c++
#include <bits/stdc++.h>

using namespace std;

const int N = 1e3 + 7;

int n, m, v, w;
int dp[N];

int main() {
    scanf("%d%d", &n, &m);
    
    for (int i = 1; i <= n; i ++) {
        scanf("%d%d", &v, &w);
        
        // 01 背包为降序， 完全背包为升序，原因见题解。
        for (int j = v; j <= m; j ++) dp[j] = max(dp[j], dp[j - v] + w);
    }
    
    printf("%d", dp[m]);
    return 0;
}
```



## [AcWing 4. 多重背包问题 I - AcWing](https://www.acwing.com/activity/content/problem/content/999/)

### 题解

> **法一：**转换为 $01$ 背包问题，将 $k$ 个相同物品，转换 $v,w$ 相同的 $k$ 个不同物品。值得注意的是，多重背包问题可以通过**二进制方法优化**，但本文不做赘述。

### CODE

```c++
#include <bits/stdc++.h>

using namespace std;

const int N = 1e3 + 7;

int n, m, s, v, w;
int dp[N];

int main() {
    scanf("%d%d", &n, &m);
    
    for (int i = 1; i <= n; i ++) {
        scanf("%d%d%d", &v, &w, &s);
        while (s --){ // 做了 s 次 01 背包
            for (int j = m; j >= v; j --) dp[j] = max(dp[j], dp[j - v] + w);
        }
        
    }
    
    printf("%d", dp[m]);
    return 0;
}
```



## [9. 分组背包问题 - AcWing题库](https://www.acwing.com/problem/content/9/)

### 题解

> **法一：**转换为 $01$ 背包问题，每次更新 $dp[j]$ 时考虑同组的不同物品。

### CODE

```c++
#include <bits/stdc++.h>

using namespace std;

const int N = 1e3 + 7;

int n, m, s;
int dp[N], v[N], w[N];

int main() {
    scanf("%d%d", &n, &m);
    
    for (int i = 1; i <= n; i ++) {
        scanf("%d", &s);
        for (int k = 0; k < s; k ++) scanf("%d%d", &v[k], &w[k]);
        
        for (int j = m; j >= 0; j --) {
            // 更新 d[j] 时，不会影响 j 之前的值，循环 s 次相当于最终只拿了一个物品
            for (int k = 0; k < s; k ++) if(j >= v[k]) dp[j] = max(dp[j], dp[j - v[k]] + w[k]);
        }
    }
    
    printf("%d", dp[m]);
    return 0;
}
```



# 二、线性 DP

## [898. 数字三角形 - AcWing题库](https://www.acwing.com/problem/content/900/)

### 题解

> **法一：**$DP$。
>
> - 状态 $dp[i][j]$ 表示从 $(1,1)$ 到 $(i,j)$ 所有方案的最大值。
> - 由于每次只能走**左下**或**右下**，所有只有 $dp[i-1][j-1]$ 和 $dp[i-1][j]$ 能走到 $dp[i][j]$。
> -  易得状态转移方程为：$dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1]) + arr[i][j]$。

### CODE

```c++
#include <bits/stdc++.h>

using namespace std;

const int N = 5e2 + 7;

int n;
int arr[N][N], dp[N][N];

int main() {
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++) {
        for (int j = 1; j <= i; j ++) {
            scanf("%d", &arr[i][j]);
        }
    }
    
    memset(dp, 0xbf, sizeof(dp));
    dp[1][1] = arr[1][1], dp[2][1] = arr[1][1] + arr[2][1], dp[2][2] = arr[1][1] + arr[2][2];
    for (int i = 3; i <= n; i ++) {
        for (int j = 1; j <= i; j ++) {
            dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1]) + arr[i][j];
        }
    }
    
    int res = 0xbfbfbfbf;
    for (int j = 1; j <= n; j ++) res = max(res, dp[n][j]);
    
    printf("%d", res);
    return 0;
}
```



## [AcWing 895. 最长上升子序列 - AcWing](https://www.acwing.com/activity/content/problem/content/1003/)

## 题解

> **法一：**$DP$。
>
> - 状态 $dp[i]$ 表示以 $arr[i]$ 为结尾的，最长上升子序列。
>
> **法二：**二分。
>
> - 维护一个子序列 $seq$，$seq$ 的长度代表**当前最长上升子序列**的长度。注，$seq$ 内的元素含义，并**不是当前最长上升子序列**，我们需要的是 $seq$ 的长度。
> - 遍历数组 $arr$：
>   - 用二分找到 $seq$ 中第一个**大于等于** $arr[i]$ 的元素位置 $j$，令 $seq[j] = arr[i]$，从而保证子序列仍然是上升的。
>   - 若 $arr[i]$ 比 $seq$ 中的所有元素都大，就将其添加到 $seq$ 的末尾。
>   - 最终，$seq$ 的长度就是最长上升子序列的长度。

## CODE

```c++
/*
法一：DP。
*/
#include <bits/stdc++.h>

using namespace std;

const int N = 1e3 + 7;

int n;
vector<int> arr(N, 0), dp(N, 1);

int main() {
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++) scanf("%d", &arr[i]);
    
    for (int i = 2; i <= n; i ++) {
        for (int j = 1; j < i; j ++) {
            if (arr[j] < arr[i]) dp[i] = max(dp[i], dp[j] + 1);
        }
    }
    
    printf("%d", *max_element(dp.begin(), dp.end()));
    return 0;
}

/*
法二：二分。
*/
#include <bits/stdc++.h>

using namespace std;

const int N = 1e3 + 7;

int n;
vector<int> arr(N, 0), seq;

int SL(int k) {
    int l = -1, r = seq.size();
    while (l + 1 < r) {
        int mid = l + r >> 1;
        if (seq[mid] < k) l = mid;
        else r = mid;
    }
    
    return r;
}

int main() {
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++) scanf("%d", &arr[i]);
    
    seq.push_back(arr[1]);
    for (int i = 2; i <= n; i ++) {
        int j = SL(arr[i]);
        if (j == seq.size()) seq.push_back(arr[i]);
        else seq[j] = arr[i];
    }
    
    printf("%d", seq.size());
    return 0;
}
```



## [897. 最长公共子序列 - AcWing题库](https://www.acwing.com/problem/content/899/)

### 题解

> **法一：**$DP$。
>
> - 状态 $dp[i][j]$  表示 $s1$ 的前 $i$ 个字母、在 $s2$ 的前 $j$ 个字母的最长公共子序列。
> - 若 $s1[i]== s2[j]$，那么 $dp[i][j]$ 的值等于 $dp[i - 1][j - 1] + 1$，即 $dp[i][j] = dp[i - 1][j - 1] + 1$。
> - 若 $s1[i]~!= s2[j]$，那么此时 $dp[i][j]$ 的值等于 $dp[i - 1][j]$ 和 $dp[i][j - 1]$ 的最大值，即 $dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])$。

### CODE

```c++
#include <bits/stdc++.h>

using namespace std;

const int N = 1e3 + 8;

int n, m;
char s1[N], s2[N];
int dp[N][N];


int main() {
    cin >> n >> m >> s1 + 1 >> s2 + 1;
    
    for (int i = 1; i <= n; i ++) {
        for (int j = 1; j <= m; j ++) {
            if (s1[i] == s2[j]) dp[i][j] = dp[i - 1][j - 1] + 1;
            else dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    
    printf("%d", dp[n][m]);
    return 0;
}
```



## [902. 最短编辑距离 - AcWing题库](https://www.acwing.com/problem/content/904/)

### 题解

> 法一：$DP$。
>
> - 状态 $dp[i][j]$ 表示将 $s1[i]$ 变为 $s2[j]$ 的最短编辑距离 ($s1$、$s2$ 下标从 $1$ 开始）。
> - 删除操作：删除 $s1[i]$ 之后，$s1[1 \sim i]$ 和 $s2[1 \sim j]$匹配，那么之前要先做到 $s1[1 \sim (i-1)]$ 和 $s2[1 \sim j]$ 匹配，故 $dp[i-1][j] + 1$。
> - 插入操作：插入 $s2[j]$ 之后，$s1[1 \sim i]$ 与 $s2[1 \sim j]$ 完全匹配，那么之前要先做到 $s1[1 \sim i]$ 和 $s2[1 \sim (j-1)]$ 匹配，故 $dp[i][j-1] + 1$。
> - 替换操作：把 $s1[i]$ 改成 $s2[j]$ 之后，$s1[1 \sim i]$ 与 $s2[1 \sim j]$ 匹配 ，那么之前要先做到 $s1[1 \sim (i-1)]$ 应该与 $s2[1 \sim (j-1)]$ 匹配，故 $dp[i-1][j-1] + 1$。若本来 $s1[i]== s2[j]$，则无需替换，即 $dp[i-1][j-1] + 0$。
>

### CODE

```c++
#include <bits/stdc++.h>

using namespace std;

const int N = 1e3 + 7;

int n, m;
char s1[N], s2[N];
int dp[N][N];

int main() {
    cin >> n >> s1 + 1 >> m >> s2 + 1;

    memset(dp, 0x3f, sizeof(dp));
    for (int i = 0; i <= n; i ++) dp[i][0] = i;
    for (int j = 0; j <= m; j ++) dp[0][j] = j;
    
    for (int i = 1; i <= n; i ++) {
        for(int j = 1; j <= m; j ++) {
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1); // 删除操作，插入操作
            
            if (s1[i] == s2[j]) dp[i][j] = min(dp[i][j], dp[i - 1][j - 1]); // 替换操作
            else dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + 1); // 替换操作

        }
    }
    
    printf("%d", dp[n][m]);
    return 0;
}
```



## [AcWing 899. 编辑距离 - AcWing](https://www.acwing.com/activity/content/problem/content/1006/)

## 题解

> 

## CODE

```c++
```

