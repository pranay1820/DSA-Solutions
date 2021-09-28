#include <bits/stdc++.h>
using namespace std;

vector<int> repeatedNumber(const vector<int> &a)
{
    int x = 0;
    int n = a.size();
    for (int i = 0; i < n; i++)
    {
        x ^= a[i];
        x ^= (i + 1);
    }
    int num = 1;
    while (1)
    {
        if (num & x)
        {
            break;
        }
        num = num << 1;
    }
    //cout<<num<<" ";
    int first = 0;
    int second = 0;
    for (int i = 0; i < n; i++)
    {
        if (a[i] & num)
        {
            first ^= a[i];
        }
        else
        {
            second ^= a[i];
        }
        if ((i + 1) & num)
        {
            first ^= (i + 1);
        }
        else
        {
            second ^= (i + 1);
        }
    }

    bool f = false;
    for (int i = 0; i < n; i++)
    {
        if (first == a[i])
        {
            f = true;
            break;
        }
    }
    vector<int> ans;
    if (f)
    {
        ans.push_back(first);
        ans.push_back(second);
    }
    else
    {
        ans.push_back(second);
        ans.push_back(first);
    }
    return ans;
}

void merge(vector<int> &nums1, int m, vector<int> &nums2, int n)
{

    int i = m - 1;
    int j = n - 1;
    int ind = n + m - 1;
    while (i >= 0 && j >= 0)
    {
        if (nums1[i] >= nums2[j])
        {
            nums1[ind] = nums1[i];
            ind--;
            i--;
        }
        else
        {
            nums1[ind] = nums2[j];
            j--;
            ind--;
        }
    }

    while (j >= 0)
    {
        nums1[ind] = nums2[j];
        j--;
        ind--;
    }
}

int maxSubArray(vector<int> &nums)
{
    int ans = nums[0];
    int so_far = nums[0];
    int n = nums.size();
    for (int i = 1; i < n; i++)
    {
        so_far = max(so_far + nums[i], nums[i]);
        ans = max(ans, so_far);
    }
    return ans;
}

// Given an array of intervals where intervals[i] = [starti, endi],
//  merge all overlapping intervals,
//   and return an array of the non-overlapping intervals that cover all the intervals in the input.
// static bool compare(vector<int> a, vector<int> b)
// {
//     if (a[0] == b[0])
//     {
//         return a[1] < b[1];
//     }
//     else
//     {
//         return a[0] < b[0];
//     }
// }

vector<vector<int>> merge(vector<vector<int>> &intervals)
{
    //sort(intervals.begin(),intervals.end(),compare);
    vector<vector<int>> ans;
    vector<int> cur = {intervals[0][0], intervals[0][1]};
    for (auto it : intervals)
    {
        if (it[0] > cur[1])
        {
            ans.push_back(cur);
            cur = it;
        }
        else
        {
            cur[0] = min(cur[0], it[0]);
            cur[1] = max(cur[1], it[1]);
        }
    }
    ans.push_back(cur);
    return ans;
}

//Given an m x n integer matrix matrix,
// if an element is 0, set its entire row and column to 0's,
// and return the matrix using constant space.
void setZeroes(vector<vector<int>> &matrix)
{
    int n = matrix.size();
    int m = matrix[0].size();
    bool f = false;
    for (int i = 0; i < n; i++)
    {
        if (matrix[i][0] == 0)
        {
            f = true;
        }
        for (int j = 1; j < m; j++)
        {
            if (matrix[i][j] == 0)
            {
                matrix[i][0] = 0;
                matrix[0][j] = 0;
            }
        }
    }

    for (int i = 1; i < n; i++)
    {
        for (int j = 1; j < m; j++)
        {
            if (matrix[0][j] == 0 || matrix[i][0] == 0)
            {
                matrix[i][j] = 0;
            }
        }
    }

    if (matrix[0][0] == 0)
    {
        for (int j = 0; j < m; j++)
        {
            matrix[0][j] = 0;
        }
    }
    if (f)
    {
        for (int i = 0; i < n; i++)
        {
            matrix[i][0] = 0;
        }
    }
}

// Implement next permutation,
//  which rearranges numbers into the lexicographically next greater permutation of numbers.
// If such an arrangement is not possible,
//  it must rearrange it as the lowest possible order (i.e., sorted in ascending order).
// The replacement must be in place and use only constant extra memory.

void nextPermutation(vector<int> &nums)
{

    int n = nums.size();
    int ind = -1;
    bool f = false;

    for (int i = n - 1; i > 0; i--)
    {
        if (nums[i] > nums[i - 1])
        {
            f = true;
            ind = i - 1;
            break;
        }
    }
    if (!f)
    {
        // reverse(nums,0,n-1);
        return;
    }

    int x = ind + 1;
    while (x < n && nums[x] > nums[ind])
    {
        x++;
    }
    x--;

    int temp = nums[ind];
    nums[ind] = nums[x];
    nums[x] = temp;

    // reverse(nums,ind+1,n-1);
}

// we can count number of inversion in an array by modifying the merge sort algorithm
// an inversion is i<j where a[i]>a[j], this algorithm works in nlog(n) time

void merge(long long arr[], long long &ans, int l, int mid, int r, int N)
{
    long long temp[N];
    int ind = l;
    int first = l;
    int second = mid + 1;
    while (first <= mid && second <= r)
    {
        if (arr[first] <= arr[second])
        {
            temp[ind++] = arr[first++];
        }
        else
        {
            temp[ind++] = arr[second++];
            ans += ((mid + 1) - first);
        }
    }
    while (first <= mid)
    {
        temp[ind++] = arr[first++];
    }
    while (second <= r)
    {
        temp[ind++] = arr[second++];
    }
    for (int i = l; i <= r; i++)
    {
        arr[i] = temp[i];
    }
}

void mergeSort(long long arr[], long long &ans, int l, int r, int N)
{
    if (l < r)
    {
        int mid = (l + r) / 2;
        mergeSort(arr, ans, l, mid, N);
        mergeSort(arr, ans, mid + 1, r, N);
        merge(arr, ans, l, mid, r, N);
    }
}

//rotate a given n*n matrix by 90deg
//here first take transpose (swap matrix[i][j] with matrix[j][i] ) of the matrix and then reverse each row
void rotate(vector<vector<int>> &matrix)
{
    int n = matrix.size();

    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            int temp = matrix[j][i];
            matrix[j][i] = matrix[i][j];
            matrix[i][j] = temp;
        }
    }

    for (int i = 0; i < n; i++)
    {
        int l = 0;
        int r = n - 1;
        while (l < r)
        {
            int temp = matrix[i][l];
            matrix[i][l] = matrix[i][r];
            matrix[i][r] = temp;
            l++;
            r--;
        }
    }
}

// Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
// Integers in each row are sorted from left to right.
// The first integer of each row is greater than the last integer of the previous row.
bool binarySearch(vector<vector<int>> mat, int target, int l, int r, int n, int m)
{
    while (r - l > 1)
    {
        int mid = (l + r) / 2;
        if (mat[mid / m][mid % m] == target)
            return true;
        else if (mat[mid / m][mid % m] > target)
        {
            r = mid;
        }
        else
        {
            l = mid;
        }
    }
    if (mat[l / m][l % m] == target || mat[r / m][r % m] == target)
        return true;
    return false;
}
bool searchMatrix(vector<vector<int>> &matrix, int target)
{
    int n = matrix.size();
    int m = matrix[0].size();
    return binarySearch(matrix, target, 0, m * n - 1, n, m);
}

// Implement pow(x, n), which calculates x raised to the power n (i.e., x^n) and x can be negative.
// This is log(n) approach 
double myPow(double x, int n)
{
    long nn = (long)abs(n);
    double ans = 1;
    while (nn > 0)
    {
        if (nn % 2)
        {
            ans *= x;
            nn--;
        }
        else
        {
            x *= x;
            nn /= 2;
        }
    }
    return n < 0 ? (double)1 / ans : ans;
}



// Given an array nums of size n, return the majority element.
// The majority element is the element that appears more than ⌊n / 2⌋ times.
// You may assume that the majority element always exists in the array
//following problem is solved using moore's voting algorithm 
int majorityElement(vector<int> &nums)
{
    int maj = -1;
    int cnt = 0;
    for (auto it : nums)
    {
        if (cnt == 0)
        {
            maj = it;
        }
        if (maj == it)
        {
            cnt++;
        }
        else
        {
            cnt--;
        }
    }
    return maj;
}



// Given an integer array of size n,
// find all elements that appear more than ⌊ n/3 ⌋ times.
vector<int> majorityElement2(vector<int> &nums)
{
    int cnt1 = 0;
    int el1 = -1;
    int cnt2 = 0;
    int el2 = -1;
    for (auto it : nums)
    {
        if (it == el1)
        {
            cnt1++;
        }
        else if (it == el2)
        {
            cnt2++;
        }
        else if (cnt1 == 0)
        {
            el1 = it;
            cnt1++;
        }
        else if (cnt2 == 0)
        {
            el2 = it;
            cnt2++;
        }
        else
        {
            cnt1--;
            cnt2--;
        }
    }
    cnt1 = 0;
    cnt2 = 0;
    for (auto it : nums)
    {
        if (it == el1)
        {
            cnt1++;
        }
        else if (it == el2)
        {
            cnt2++;
        }
    }
    vector<int> ans;
    if (cnt1 > nums.size() / 3)
    {
        ans.push_back(el1);
    }
    if (cnt2 > nums.size() / 3)
    {
        ans.push_back(el2);
    }
    return ans;
}


// Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:
// 0 <= a, b, c, d < n
// a, b, c, and d are distinct.
// nums[a] + nums[b] + nums[c] + nums[d] == target
// You may return the answer in any order.
vector<vector<int>> fourSum(vector<int> &nums, int target)
{
    int n = nums.size();
    vector<vector<int>> ans;
    sort(nums.begin(), nums.end());
    int prev1 = nums[0] - 1;
    for (int i = 0; i < n;)
    {
        while (i < n && nums[i] == prev1)
        {
            i++;
        }
        if (i >= n)
        {
            break;
        }
        prev1 = nums[i];

        int prev2 = nums[i] - 1;

        for (int j = i + 1; j < n;)
        {
            while (j < n && nums[j] == prev2)
            {
                j++;
            }

            if (j >= n)
            {
                break;
            }

            prev2 = nums[j];

            int newTarget = target - (nums[i] + nums[j]);
            int l = j + 1;
            int r = n - 1;
            while (l < r)
            {
                if (nums[l] + nums[r] < newTarget)
                {
                    l++;
                }
                else if (nums[l] + nums[r] > newTarget)
                {
                    r--;
                }
                else
                {
                    vector<int> temp;
                    temp.push_back(nums[i]);
                    temp.push_back(nums[j]);
                    temp.push_back(nums[l]);
                    temp.push_back(nums[r]);
                    ans.push_back(temp);
                    int l1 = l;
                    while (l1 < n && nums[l1] == nums[l])
                    {
                        l1++;
                    }
                    l = l1;
                    int r1 = r;
                    while (r1 >= 0 && nums[r1] == nums[r])
                    {
                        r1--;
                    }
                    r = r1;
                }
            }
        }
    }
    return ans;
}



// Given an unsorted array of integers nums, 
// return the length of the longest consecutive elements sequence.
// You must write an algorithm that runs in O(n) time.
// Input: nums = [100,4,200,1,3,2]
// Output: 4
// Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4
int longestConsecutive(vector<int> &nums)
{
    unordered_set<int> st;
    for (auto it : nums)
    {
        st.insert(it);
    }

    int ans = 0;
    for (int it : nums)
    {
        if (st.find(it - 1) == st.end())
        {
            int cnt = 0;
            int cur = it;
            while (st.find(cur) != st.end())
            {
                cur++;
                cnt++;
            }
            ans = max(ans, cnt);
        }
    }
    return ans;
}


// Given an array of integers A and an integer B.
// Find the total number of subarrays having bitwise XOR of all elements equals to B.
int NumberOfSubArraysWithGivenXor(vector<int> &A, int B) {
    unordered_map<int,int>mp;
    int xor1=0;
    mp[0]=1;
    int ans=0;
    for(auto it:A){
        xor1^=it;
        mp[xor1]++;
        ans+=mp[xor1^B];
    }
    return ans;
}



//Given a string s, find the length of the longest substring without repeating characters.
int lengthOfLongestSubstring(string s)
{
    int cnt[256] = {0};
    int n = s.length();
    int j = 0;
    int ans = 0;
    int i;
    for (i = 0; i < n && j < n; i++)
    {
        while (j < n && cnt[s[j]] < 1)
        {
            cnt[s[j]]++;
            j++;
        }

        ans = max(ans, j - i);
        cnt[s[i]]--;
    }
    return ans;
}


// Given the heads of two singly linked-lists headA and headB,
// return the node at which the two lists intersect.
// If the two linked lists have no intersection at all, return null
struct ListNode {
     int val;
     ListNode *next;
     ListNode(int x) : val(x), next(NULL) {}
};
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB)
{
    int flag = 0;
    ListNode *a = headA;
    ListNode *b = headB;
    while (flag <= 2)
    {
        if (a == b)
            return a;
        a = a->next;
        b = b->next;
        if (a == NULL)
        {
            flag++;
            a = headB;
        }
        if (b == NULL)
        {
            flag++;
            b = headA;
        }
    }
    return NULL;
}

//Given head, the head of a linked list, determine if the linked list has a cycle in it.
bool hasCycle(ListNode *head)
{
    ListNode *fast = head;
    ListNode *slow = head;
    while (fast && fast->next)
    {
        slow = slow->next;
        fast = fast->next->next;
        if (fast == slow)
            return true;
    }
    return false;
}


//Given the head of a singly linked list, return true if it is a palindrome.
bool isPalindrome(ListNode *head)
{
    ListNode *fast = head;
    ListNode *slow = head;
    while (fast->next && fast->next->next)
    {
        fast = fast->next->next;
        slow = slow->next;
    }

    slow->next = //reverse(slow->next);
    slow = slow->next;
    fast = head;
    while (slow != NULL)
    {
        if (slow->val != fast->val)
        {
            return false;
        }
        fast = fast->next;
        slow = slow->next;
    }
    return true;
}


//Given the head of a linked list, rotate the list to the right by k places.
ListNode *rotateRight(ListNode *head, int k)
{
    if (!head)
        return head;
    int size = 0;
    ListNode *temp = head;
    ListNode *prev = NULL;
    while (temp)
    {
        prev = temp;
        temp = temp->next;
        size++;
    }
    k %= size;
    temp = head;
    for (int i = 1; i < size - k; i++)
    {
        temp = temp->next;
    }
    prev->next = head;
    head = temp->next;
    temp->next = NULL;
    return head;
}


//Given a linked list, return the node where the cycle begins.
//If there is no cycle, return null
ListNode *detectCycleAndReturnStartingNode(ListNode *head)
{
    ListNode *slow = head;
    ListNode *fast = head;
    while (fast && fast->next)
    {
        slow = slow->next;
        fast = fast->next->next;
        if (fast == slow)
        {
            break;
        }
    }

    if (!fast || !fast->next)
    {
        return NULL;
    }
    fast = head;
    while (fast != slow)
    {
        fast = fast->next;
        slow = slow->next;
    }
    return slow;
}

// Given a Linked List of size N, where every node represents a sub-linked-list and contains two pointers:
// (i) a next pointer to the next node,
// (ii) a bottom pointer to a linked list where this node is head.
// Each of the sub-linked-list is in sorted order.
// Flatten the Link List such that all the nodes appear in a single level while maintaining the sorted order. 
// Note: The flattened list will be printed using the bottom pointer instead of next pointer.
Node *flatten(Node *root)
{
  if(root->next==NULL)
  return root;
  
  root->next=flatten(root->next);
  root=merge(root,root->next);
  return root;
}


//Given n non-negative integers representing an elevation map where the width of each bar is 1,
//compute how much water it can trap after raining.
int trap(vector<int> &height)
{
    int ans = 0;
    int n = height.size();
    int l = 0;
    int r = n - 1;
    int lmx = 0;
    int rmx = 0;
    while (l <= r)
    {
        if (height[l] <= height[r])
        {
            if (height[l] >= lmx)
            {
                lmx = height[l];
            }
            else
            {
                ans += lmx - height[l];
            }
            l++;
        }
        else
        {
            if (rmx > height[r])
            {
                ans += rmx - height[r];
            }
            else
            {
                rmx = height[r];
            }
            r--;
        }
    }
    return ans;
}



// A linked list of length n is given such that each node contains an additional random pointer,
// which could point to any node in the list, or null.
// Construct a deep copy of the list.
// The deep copy should consist of exactly n brand new nodes,
// where each new node has its value set to the value of its corresponding original node.
// Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. 
// None of the pointers in the new list should point to nodes in the original list.
Node *copyRandomList(Node *head)
{
    Node *temp = head;
    while (temp)
    {
        Node *d = new Node(temp->val);
        d->next = temp->next;
        temp->next = d;
        temp = temp->next->next;
    }
    temp = head;
    while (temp)
    {
        if (temp->random)
            temp->next->random = temp->random->next;
        else
            temp->next->random = NULL;
        temp = temp->next->next;
    }

    temp = head;
    Node *toReturn = NULL;
    while (temp)
    {
        Node *d = temp->next;
        temp->next = d->next;
        if (temp->next)
        {
            d->next = temp->next->next;
        }
        else
        {
            d->next = NULL;
        }
        temp = temp->next;
        if (!toReturn)
        {
            toReturn = d;
        }
    }

    return toReturn;
}


// Given arrival and departure times of all trains that reach a railway station.
//  Find the minimum number of platforms required for the railway station so that no train is kept waiting.
// Consider that all the trains arrive on the same day and leave on the same day. 
// Arrival and departure time can never be the same for a train but we can have arrival time of one train equal to departure time of the other.
//  At any given instance of time, same platform can not be used for both departure of a train and arrival of another train. 
// In such cases, we need different platforms.
int findPlatform(int arr[], int dep[], int n)
{
    // Your code here
    int ans = 0;
    sort(arr, arr + n);
    sort(dep, dep + n);

    int l = 0;
    int r = 0;
    int count = 0;
    while (l < n)
    {
        if (arr[l] <= dep[r])
        {
            count++;
            l++;
        }
        else
        {
            count--;
            r++;
        }
        ans = max(ans, count);
    }
    return ans;
}


// Given an array of distinct integers candidates and a target integer target, 
// return a list of all unique combinations of candidates where the chosen numbers sum to target.
// You may return the combinations in any order.
// The same number may be chosen from candidates an unlimited number of times.
// Two combinations are unique if the frequency of at least one of the chosen numbers is different.
void solve(vector<int> &c, int target, vector<vector<int>> &ans, vector<int> temp, int index)
{
    if (target == 0)
    {
        ans.push_back(temp);
        return;
    }
    if (index >= c.size())
    {
        return;
    }
    if (target < 0)
    {
        return;
    }

    solve(c, target, ans, temp, index + 1);
    temp.push_back(c[index]);
    solve(c, target - c[index], ans, temp, index);
}
vector<vector<int>> combinationSum(vector<int> &candidates, int target)
{
    vector<vector<int>> ans;
    vector<int> temp;
    solve(candidates, target, ans, temp, 0);
    return ans;
}


// Given a collection of candidate numbers (candidates) and a target number (target),
// find all unique combinations in candidates where the candidate numbers sum to target.
// Each number in candidates may only be used once in the combination.
void solve(vector<int> &c, int target, vector<vector<int>> &ans, vector<int> temp, int index, int n)
{
    if (target < 0)
    {
        return;
    }
    if (target == 0)
    {
        ans.push_back(temp);
        return;
    }

    for (int i = index; i < n;)
    {
        temp.push_back(c[i]);
        solve(c, target - c[i], ans, temp, i + 1, n);
        temp.pop_back();
        int j = i;
        while (j < n && c[j] == c[i])
        {
            j++;
        }
        i = j;
    }
}

vector<vector<int>> combinationSum2(vector<int> &c, int target)
{
    sort(c.begin(), c.end());
    vector<vector<int>> ans;
    vector<int> temp;
    solve(c, target, ans, temp, 0, c.size());
    return ans;
}


//Generate permutations without using extra space using backtracking
//here every index is swapped with current index to generate all permutations
void solve(vector<int> &nums, vector<vector<int>> &ans, vector<int> temp, int n, int index)
{
    if (index == n)
    {
        ans.push_back(temp);
        return;
    }

    for (int i = index; i < n; i++)
    {
        swap(nums, index, i);
        temp.push_back(nums[index]);
        solve(nums, ans, temp, n, index + 1);
        temp.pop_back();
        swap(nums, index, i);
    }
}


// sudoku solution must satisfy all of the following rules:
// Each of the digits 1-9 must occur exactly once in each row.
// Each of the digits 1-9 must occur exactly once in each column.
// Each of the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
// The '.' character indicates empty cells.
bool valid(vector<vector<char>> &board, int row, int col, int c)
{
    for (int i = 0; i < 9; i++)
    {
        if (board[row][i] == c + '0')
            return false;

        if (board[i][col] == c + '0')
            return false;

        if (board[(row / 3) * 3 + i / 3][(col / 3) * 3 + i % 3] == c + '0')
            return false;
    }
    return true;
}

bool solve(vector<vector<char>> &board)
{
    int row = -1;
    int col = -1;
    for (int i = 0; i < 9 && row == -1; i++)
    {
        for (int j = 0; j < 9 && row == -1; j++)
        {
            if (board[i][j] == '.')
            {
                row = i;
                col = j;
            }
        }
    }

    if (row == -1 && col == -1)
    {
        return true;
    }

    for (int i = 1; i <= 9; i++)
    {
        if (valid(board, row, col, i))
        {
            board[row][col] = i + '0';
            if (solve(board))
            {
                return true;
            }
            board[row][col] = '.';
        }
    }
    return false;
}

// Consider a rat placed at (0, 0) in a square matrix of order N * N.
// It has to reach the destination at (N - 1, N - 1).
// Find all possible paths that the rat can take to reach from source to destination.
// The directions in which the rat can move are 'U'(up), 'D'(down), 'L' (left), 'R' (right).
// Value 0 at a cell in the matrix represents that it is blocked and rat cannot move to it
// while value 1 at a cell in the matrix represents that rat can be travel through it.
void solve(vector<vector<int>> &m, int n, vector<string> &ans, string curr, int row, int col)
{

    if (row == n - 1 && col == n - 1)
    {
        ans.push_back(curr);
        return;
    }

    if (row - 1 >= 0 && m[row - 1][col] == 1)
    {
        m[row][col] = 0;
        solve(m, n, ans, curr + "U", row - 1, col);
        m[row][col] = 1;
    }

    if (row + 1 < n && m[row + 1][col] == 1)
    {
        m[row][col] = 0;
        solve(m, n, ans, curr + "D", row + 1, col);
        m[row][col] = 1;
    }
    if (col - 1 >= 0 && m[row][col - 1] == 1)
    {
        m[row][col] = 0;
        solve(m, n, ans, curr + "L", row, col - 1);
        m[row][col] = 1;
    }
    if (col + 1 < n && m[row][col + 1] == 1)
    {
        m[row][col] = 0;
        solve(m, n, ans, curr + "R", row, col + 1);
        m[row][col] = 1;
    }
}


// Given an undirected graph and an integer M.
// The task is to determine if the graph can be colored with at most M colors such that no two adjacent vertices of the graph are colored with the same color.
// Here coloring of a graph means the assignment of colors to all vertices.
// Print 1 if it is possible to colour vertices and 0 otherwise.

bool solve(int color[],bool graph[101][101],int m,int V,int curr){
    if(curr==V){
        return true;
    }
    
    for(int i=1;i<=m;i++){
        color[curr]=i;
        if(valid(color,graph,curr,V) && solve(color,graph,m,V,curr+1)){
            return true;
        }
        color[curr]=0;
    }
    return false;
}


// This implementation of KMP pattern searching algorithm 
// Time complexity is O(n)
void lps1(string pat, int lps[])
{
    int len = 0;
    int n = pat.length();
    int i = 1;
    lps[0] = 0;
    while (i < n)
    {
        if (pat[i] == pat[len])
        {
            lps[i++] = ++len;
        }
        else
        {
            if (len == 0)
            {
                lps[i++] = 0;
            }
            else
            {
                len = lps[len - 1];
            }
        }
    }
}

vector<int> search(string pat, string txt)
{
    //code hee.
    vector<int> ans;
    int i = 0;
    int j = 0;
    int n = txt.length();
    int m = pat.length();

    int lps[m];
    lps1(pat, lps);

    while (i < n)
    {
        if (pat[j] == txt[i])
        {
            i++;
            j++;
        }
        if (j == m)
        {
            ans.push_back(i - j + 1);
            j = lps[j - 1];
        }
        if (pat[j] != txt[i])
        {
            if (j == 0)
            {
                i++;
            }
            else
            {
                j = lps[j - 1];
            }
        }
    }
    return ans;
}

//You are given a sorted array consisting of only integers where every element appears exactly twice,
//except for one element which appears exactly once.
//Find this single element that appears only once.

int singleNonDuplicate(vector<int> &nums)
{
    int l = -1;
    int r = nums.size() - 1;
    while (r - l > 1)
    {
        int mid = (l + r) / 2;
        if ((mid % 2 && nums[mid - 1] == nums[mid]) || (mid % 2 == 0 && nums[mid + 1] == nums[mid]))
        {
            l = mid;
        }
        else
        {
            r = mid;
        }
    }
    return nums[r];
}

//Given a sorted rotated array find if the target
//element is present in the given array
//expected time is O(log(n)) using binary search
int search(vector<int> &nums, int target)
{
    int l = 0;
    int r = nums.size() - 1;

    while (l <= r)
    {
        int mid = (l + r) / 2;
        if (nums[mid] == target)
        {
            return mid;
        }

        if (nums[mid] >= nums[l])
        {
            if (nums[mid] < target || nums[l] > target)
            {
                l = mid + 1;
            }
            else
            {
                r = mid - 1;
            }
        }
        else
        {
            if (nums[mid] > target || nums[r] < target)
            {
                r = mid - 1;
            }
            else
            {
                l = mid + 1;
            }
        }
    }
    return -1;
}


//Find median of given 2 sorted arrays 
//expected time complexity is O(log(n+m))
double findMedianSortedArrays(vector<int> &nums1, vector<int> &nums2)
{
    int n = nums1.size();
    int m = nums2.size();
    int ne = (n + m + 1) / 2;
    int l = 0;
    int r = n;

    int el1 = 0, el2 = 0;
    while (l <= r)
    {
        int mid = (l + r) / 2;
        int rem = ne - mid;
        if (rem < 0)
        {
            r = mid - 1;
            continue;
        }

        if (rem > m)
        {
            l = mid + 1;
            continue;
        }

        int x1 = -1e7, x2 = -1e7, y1 = 1e7, y2 = 1e7;
        if (mid > 0 && mid <= n && n > 0)
        {
            x1 = nums1[mid - 1];
        }
        if (rem > 0 && rem <= m && m > 0)
        {
            x2 = nums2[rem - 1];
        }
        if (mid < n && mid >= 0 && n > 0)
        {
            y1 = nums1[mid];
        }
        if (rem < m && rem >= 0 && m > 0)
        {
            y2 = nums2[rem];
        }

        if (x1 <= y2 && x2 <= y1)
        {
            el1 = max(x1, x2);
            el2 = min(y1, y2);
            break;
        }

        if (x1 > y2)
        {
            r = mid - 1;
        }
        else
        {
            l = mid + 1;
        }
    }

    if ((n + m) % 2)
    {
        return el1;
    }
    return double(el1 + el2) / double(2);
}

// Given an array of integers A of size N and an integer B.
// College library has N bags,the ith book has A[i] number of pages.
// You have to allocate books to B number of students so that maximum number of pages alloted to a student is minimum.
// A book will be allocated to exactly one student.
// Each student has to be allocated at least one book.
// Allotment should be in contiguous order, for example: A student cannot be allocated book 1 and book 3, skipping book 2.
// Calculate and return that minimum possible number.
bool valid(int pages,vector<int>&a,int b){
    int x=0;
    int curPages=0;
    for(auto it:a){
        if(it>pages){
            return false;
        }
        if(curPages+it>pages){
            x++;
            curPages=it;
        }else{
            curPages+=it;
        }
    }
    return x<b;
}
int books(vector<int> &A, int B) {
    if(B>A.size()){
        return -1;
    }
    int l=0;
    int r=1e10+5;
    
    while(r-l>1){
        int mid=(l+r)/2;
        if(valid(mid,A,B)){
            r=mid;
        }else{
            l=mid;
        }
    }
    return r;
}


// Given an input string s, reverse the order of the words.
// A word is defined as a sequence of non-space characters. The words in s will be separated by at least one space.
// Return a string of the words in reverse order concatenated by a single space.
string reverseWords(string s)
{
    int n = s.length();
    int l = 0;
    int r = n - 1;
    while (l < r)
    {
        char temp = s[l];
        s[l] = s[r];
        s[r] = temp;
        l++;
        r--;
    }
    int red = 0;
    for (int i = 0; i < n - red;)
    {
        int j = i;
        while (j < n - red && s[j] != ' ')
        {
            j++;
        }
        int l = i;
        int r = j - 1;
        while (l < r)
        {
            char temp = s[l];
            s[l] = s[r];
            s[r] = temp;
            l++;
            r--;
        }
        int cnt = 1;
        j++;
        while (j < n - red && s[j] == ' ')
        {
            j++;
            cnt++;
        }

        if (cnt - 1 > 0)
        {
            red += cnt - 1;
            s.erase(j - cnt, cnt - 1);
        }

        i = j - cnt + 1;
    }
    n = s.length();
    l = 0;
    while (l < n && s[l] == ' ')
    {
        l++;
    }
    r = n - 1;
    while (r >= 0 && s[r] == ' ')
    {
        r--;
    }
    return s.substr(l, r - l + 1);
}

//Given a string s, return the longest palindromic substring in s
string longestPalindrome(string s)
{
    int n = s.length();
    bool dp[n][n];
    memset(dp, false, sizeof(dp));
    int len = 0;
    int l = 0;
    for (int i = n - 1; i >= 0; i--)
    {
        for (int j = n - 1; j >= i; j--)
        {
            if (i == j)
            {
                dp[i][j] = true;
            }
            else if (i + 1 == j)
            {
                dp[i][j] = s[i] == s[j];
            }
            else
            {
                dp[i][j] = (s[i] == s[j] && dp[i + 1][j - 1]);
            }

            if (dp[i][j] && j - i + 1 > len)
            {
                len = j - i + 1;
                l = i;
            }
        }
    }

    return s.substr(l, len);
}


// Write a function to find the longest common prefix string amongst an array of strings.
// If there is no common prefix, return an empty string ""
string longestCommonPrefix(vector<string> &s)
{
    int r = s[0].length();
    int n = s.size();
    for (int i = 1; i < n; i++)
    {
        int m = s[i].length();
        for (int j = 0; j < min(m, r); j++)
        {
            if (s[0][j] != s[i][j])
            {
                r = j;
                break;
            }
        }
        r = min(m, r);
    }

    return s[0].substr(0, r);
}


// You are given an array of integers nums, 
// there is a sliding window of size k which is moving from the very left of the array to the very right.
// You can only see the k numbers in the window. 
// Each time the sliding window moves right by one position
vector<int> maxSlidingWindow(vector<int> &nums, int k)
{
    deque<int> dq;
    for (int i = 0; i < k; i++)
    {
        while (!dq.empty() && nums[dq.back()] < nums[i])
        {
            dq.pop_back();
        }

        dq.push_back(i);
    }
    vector<int> ans;
    ans.push_back(nums[dq.front()]);
    int n = nums.size();
    for (int i = k; i < n; i++)
    {
        if (!dq.empty() && dq.front() < i - (k - 1))
        {
            dq.pop_front();
        }
        while (!dq.empty() && nums[dq.back()] < nums[i])
        {
            dq.pop_back();
        }

        dq.push_back(i);
        ans.push_back(nums[dq.front()]);
    }
    return ans;
}

//sort a stack
void sortStack(stack<int>s)
{
    stack<int>temp;
    while(!s.empty()){
        int x=s.top();
        s.pop();
        while(!temp.empty() && x>temp.top()){
            s.push(temp.top());
            temp.pop();
        }
        
        temp.push(x);
    }
    while(!temp.empty()){
        s.push(temp.top());
        temp.pop();
    }
}


// A celebrity is a person who is known to all but does not know anyone at a party.
// If you go to a party of N people,
// find if there is a celebrity in the party or not.
// A square NxN matrix M[][] is used to represent people at the party such that if an element of row i and column j
// is set to 1 it means ith person knows jth person. Here M[i][i] will always be 0
int celebrity(vector<vector<int>> &M, int n)
{
    // code here
    int first = 0;
    int second = 1;
    for (int i = 1; i < n; i++)
    {
        if (M[first][second])
        {
            first = second;
            second = i + 1;
        }
        else
        {
            second = i + 1;
        }
    }

    int count = 0;
    for (int r = 0; r < n; r++)
    {
        if (M[r][first])
        {
            count++;
        }
        if (M[first][r])
        {
            count--;
        }
    }

    return count == n - 1 ? first : -1;
}

//Given an array of integers heights representing the histogram's bar height where the width of each bar is 1,
//return the area of the largest rectangle in the histogram.
int largestRectangleArea(vector<int> &height)
{
    int n = height.size();
    stack<int> st;
    int ans = 0;
    for (int i = 0; i < n; i++)
    {
        while (!st.empty() && height[st.top()] >= height[i])
        {
            int h = height[st.top()];
            st.pop();
            int w = i;
            if (!st.empty())
            {
                w = i - st.top() - 1;
            }
            ans = max(ans, w * h);
        }
        st.push(i);
    }

    while (!st.empty())
    {
        int h = height[st.top()];
        st.pop();
        int w = n;
        if (!st.empty())
        {
            w = n - st.top() - 1;
        }
        ans = max(ans, h * w);
    }

    return ans;
}

int main()
{

    return 0;
} 
