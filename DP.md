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



# 二、[898. 数字三角形 - AcWing题库](https://www.acwing.com/problem/content/900/)

