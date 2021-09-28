#include<bits/stdc++.h>
using namespace std;

struct TreeNode
{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

vector<int> inorderTraversal(TreeNode *root)
{
    vector<int> ans;
    stack<TreeNode *> st;

    while (!st.empty() || root)
    {
        while (root)
        {
            st.push(root);
            root = root->left;
        }

        root = st.top();
        st.pop();
        ans.push_back(root->val);
        root = root->right;
    }
    return ans;
}

vector<int> preorderTraversal(TreeNode *root)
{
    vector<int> ans;
    if (!root)
    {
        return ans;
    }
    stack<TreeNode *> st;
    st.push(root);
    while (!st.empty())
    {
        root = st.top();
        st.pop();
        ans.push_back(root->val);
        if (root->right)
        {
            st.push(root->right);
        }
        if (root->left)
        {
            st.push(root->left);
        }
    }
    return ans;
}

void leftview(Node *root, int level, int &mxlevel, vector<int> &ans)
{
    if (root == NULL)
        return;
    if (level > mxlevel)
    {
        ans.push_back(root->data);
        mxlevel = level;
    }
    leftview(root->left, level + 1, mxlevel, ans);

    leftview(root->right, level + 1, mxlevel, ans);
}

vector<int> postorderTraversal(TreeNode *root)
{
    vector<int> ans;
    if (!root)
    {
        return ans;
    }
    stack<TreeNode *> st;
    stack<int> st1;
    st.push(root);
    while (!st.empty())
    {
        root = st.top();
        st.pop();
        st1.push(root->val);
        if (root->left)
        {
            st.push(root->left);
        }
        if (root->right)
        {
            st.push(root->right);
        }
    }
    while (!st1.empty())
    {
        ans.push_back(st1.top());
        st1.pop();
    }
    return ans;
}

vector<int> bottomView(Node *root)
{
    unordered_map<int, int> mp;
    queue<pair<Node *, int>> q;
    q.push({root, 0});
    int mn = INT_MAX;
    int mx = INT_MIN;
    while (!q.empty())
    {
        pair<Node *, int> cur = q.front();
        q.pop();
        mp[cur.second] = cur.first->data;
        mn = min(mn, cur.second);
        mx = max(mx, cur.second);
        if (cur.first->left)
        {
            q.push({cur.first->left, cur.second - 1});
        }
        if (cur.first->right)
        {
            q.push({cur.first->right, cur.second + 1});
        }
    }

    vector<int> ans;
    for (int i = mn; i <= mx; i++)
    {
        ans.push_back(mp[i]);
    }
    return ans;
}

vector<vector<int>> levelOrder(TreeNode *root)
{

    vector<vector<int>> ans;
    if (root == NULL)
    {
        return ans;
    }
    vector<int> temp;
    queue<TreeNode *> q;
    q.push(root);
    q.push(NULL);
    while (!q.empty())
    {
        TreeNode *cur = q.front();
        q.pop();
        if (cur == NULL)
        {
            ans.push_back(temp);
            temp.clear();
            if (!q.empty())
                q.push(NULL);
        }
        else
        {
            temp.push_back(cur->val);
            if (cur->left)
            {
                q.push(cur->left);
            }
            if (cur->right)
            {
                q.push(cur->right);
            }
        }
    }

    return ans;
}

//Using deque
vector<vector<int>> zigzagLevelOrder(TreeNode *root)
{
    vector<vector<int>> ans;
    if (root == NULL)
    {
        return ans;
    }
    vector<int> temp;
    deque<TreeNode *> q;
    q.push_back(root);
    q.push_back(NULL);
    bool flag = 0;
    while (!q.empty())
    {
        if (!flag)
        {
            TreeNode *cur = q.front();
            q.pop_front();

            if (cur == NULL)
            {
                ans.push_back(temp);
                temp.clear();
                flag = 1 - flag;
                if (!q.empty())
                {
                    q.push_front(NULL);
                }
            }
            else
            {
                temp.push_back(cur->val);
                if (cur->left)
                {
                    q.push_back(cur->left);
                }
                if (cur->right)
                {
                    q.push_back(cur->right);
                }
            }
        }
        else
        {
            TreeNode *cur = q.back();
            q.pop_back();

            if (cur == NULL)
            {
                ans.push_back(temp);
                temp.clear();
                flag = 1 - flag;
                if (!q.empty())
                {
                    q.push_back(NULL);
                }
            }
            else
            {
                temp.push_back(cur->val);
                if (cur->right)
                {
                    q.push_front(cur->right);
                }
                if (cur->left)
                {
                    q.push_front(cur->left);
                }
            }
        }
    }

    return ans;
}

bool isSameTree(TreeNode *p, TreeNode *q)
{
    if (!p && !q)
    {
        return true;
    }
    if ((p && !q) || (!p && q) || p->val != q->val)
    {
        return false;
    }
    queue<TreeNode *> q1;
    queue<TreeNode *> q2;
    q1.push(p);
    q2.push(q);

    while (!q1.empty() || !q2.empty())
    {
        if ((q1.empty() && !q2.empty()) || (!q1.empty() && q2.empty()))
        {
            return true;
        }
        TreeNode *cur1 = q1.front();
        q1.pop();
        TreeNode *cur2 = q2.front();
        q2.pop();

        if ((!cur1->left && cur2->left) || (cur1->left && !cur2->left) || ((cur1->left && cur2->left) && cur1->left->val != cur2->left->val))
        {
            return false;
        }
        if ((!cur1->right && cur2->right) || (cur1->right && !cur2->right) || ((cur1->right && cur2->right) && cur1->right->val != cur2->right->val))
        {
            return false;
        }

        if (cur1->left && cur2->left)
        {
            q1.push(cur1->left);
            q2.push(cur2->left);
        }
        if (cur1->right && cur2->right)
        {
            q1.push(cur1->right);
            q2.push(cur2->right);
        }
    }

    return true;
}

pair<bool, bool> path(TreeNode *root, TreeNode *p, TreeNode *q, TreeNode *&ans)
{
    if (root == NULL)
    {
        return {false, false};
    }
    bool f1 = false, f2 = false;
    if (root == p)
    {
        f1 = true;
    }
    if (root == q)
    {
        f2 = true;
    }

    pair<bool, bool> p1 = path(root->left, p, q, ans);
    pair<bool, bool> p2 = path(root->right, p, q, ans);
    if (ans == NULL && (p1.first || p2.first || f1) && (p1.second || p2.second || f2))
    {
        ans = root;
    }
    return {p1.first || p2.first || f1, p2.second || p1.second || f2};
}

TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q)
{

    TreeNode *ans = NULL;
    path(root, p, q, ans);
    return ans;
}

TreeNode *build(vector<int> &preorder, vector<int> &inorder, int l, int r, int ll, int rr)
{
    if (l == r)
    {
        return new TreeNode(preorder[l]);
    }
    if (l > r)
    {
        return NULL;
    }

    TreeNode *root = new TreeNode(preorder[l]);
    int partition;
    for (int i = ll; i <= rr; i++)
    {
        if (inorder[i] == preorder[l])
        {
            partition = i;
            break;
        }
    }

    root->left = build(preorder, inorder, l + 1, l + (partition - ll), ll, partition - 1);
    root->right = build(preorder, inorder, l + (partition - ll) + 1, r, partition + 1, rr);
    return root;
}

TreeNode *build(vector<int> &inorder, vector<int> &postorder, int l, int r, int ll, int rr)
{
    if (l == r)
    {
        return new TreeNode(postorder[l]);
    }

    if (l > r)
    {
        return NULL;
    }

    int partition;

    for (int i = ll; i <= rr; i++)
    {
        if (inorder[i] == postorder[r])
        {
            partition = i;
            break;
        }
    }

    return new TreeNode(
        postorder[r],
        build(inorder, postorder, l, (l + (partition - ll) - 1), ll, partition - 1),
        build(inorder, postorder, l + (partition - ll), r - 1, partition + 1, rr));
}

bool isSymmetric(TreeNode *root)
{

    queue<TreeNode *> q1;
    queue<TreeNode *> q2;
    q1.push(root);
    q2.push(root);
    while (!q1.empty())
    {
        TreeNode *first = q1.front();
        q1.pop();
        TreeNode *second = q2.front();
        q2.pop();

        if (first->val != second->val)
        {
            return false;
        }

        if ((!first->left && second->right) || (first->left && !second->right) || (!first->right && second->left) || (first->right && !second->left))
        {
            return false;
        }

        if (first->left && second->right)
        {
            q1.push(first->left);
            q2.push(second->right);
        }

        if (first->right && second->left)
        {
            q1.push(first->right);
            q2.push(second->left);
        }
    }

    return true;
}


// Given the root of a binary tree, flatten the tree into a "linked list":
// The "linked list" should use the same TreeNode class 
// where the right child pointer points to the next node in the list and the left child pointer is always null.
// The "linked list" should be in the same order as a pre-order traversal of the binary tree.

pair<TreeNode *, TreeNode *> solve(TreeNode *root)
{
    if (!root->left && !root->right)
        return {root, root};

    if (!root->left && root->right)
    {
        pair<TreeNode *, TreeNode *> p = solve(root->right);
        root->right = p.first;
        return {root, p.second};
    }
    else if (root->left && !root->right)
    {
        pair<TreeNode *, TreeNode *> p = solve(root->left);
        root->left = NULL;
        root->right = p.first;
        return {root, p.second};
    }
    else
    {
        pair<TreeNode *, TreeNode *> p1 = solve(root->left);
        pair<TreeNode *, TreeNode *> p2 = solve(root->right);

        root->left = NULL;
        root->right = p1.first;
        p1.second->right = p2.first;
        return {root, p2.second};
    }
}

void flatten(TreeNode *root)
{
    if (root)
        solve(root);
}

int solve(TreeNode *root, int &ans)
{
    if (root == NULL)
    {
        return 0;
    }

    int left = solve(root->left, ans);
    int right = solve(root->right, ans);

    ans = max(ans, left + right + root->val);
    ans = max(ans, root->val);
    ans = max(ans, max(root->val + left, root->val + right));
    return max(root->val, max(left, right) + root->val);
}
 
int maxPathSum(TreeNode *root)
{
    int ans = -1e4;
    solve(root, ans);
    return ans;
}
