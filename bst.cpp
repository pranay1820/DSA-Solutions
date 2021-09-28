
pair<int, int> solve(Node *root, bool &flag)
{
    if (root == NULL)
    {
        return {INT_MAX, INT_MIN};
    }

    pair<int, int> p1 = solve(root->left, flag);
    pair<int, int> p2 = solve(root->right, flag);
    if (p1.second >= root->data || p2.first <= root->data || (root->left && root->left->data == root->data) || (root->right && root->right->data == root->data))
    {
        flag = false;
    }

    return {min(p1.first, min(p2.first, root->data)), max(root->data, max(p2.second, p1.second))};
}

//Function to check whether a Binary Tree is BST or not.
bool isBST(Node *root)
{
    bool flag = true;
    solve(root, flag);
    return flag;
}


//program to find lcs of two nodes in a BST
pair<bool, bool> lcs(TreeNode *&ans, TreeNode *root, TreeNode *p, TreeNode *q)
{
    if (root == NULL)
    {
        return {false, false};
    }
    bool flag1 = false;
    bool flag2 = false;
    if (root->val == p->val)
    {
        flag1 = true;
    }
    if (root->val == q->val)
    {
        flag2 = true;
    }
    pair<bool, bool> p1, p2;
    if (p->val < root->val)
    {
        p1 = lcs(ans, root->left, p, q);
    }
    else
    {
        p1 = lcs(ans, root->right, p, q);
    }
    if (q->val < root->val)
    {
        p2 = lcs(ans, root->left, p, q);
    }
    else
    {
        p2 = lcs(ans, root->right, p, q);
    }

    if ((p2.first || p1.first || flag1) && (p1.second || p2.second || flag2) && ans == NULL)
    {
        ans = root;
    }
    return {flag1 || p1.first || p2.first, flag2 || p2.second || p1.second};
}

TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q)
{
    TreeNode *ans = NULL;
    lcs(ans, root, p, q);
    return ans;
}

// This function finds predecessor and successor of key in BST.
// It sets pre and suc as predecessor and successor respectively
void findPreSuc(Node *root, Node *&pre, Node *&suc, int key)
{
    Node *temp = root;
    while (temp)
    {
        if (temp->key == key)
        {
            Node *x = temp->left;
            while (x)
            {
                pre = x;
                x = x->right;
            }

            x = temp->right;
            while (x)
            {
                suc = x;
                x = x->left;
            }

            return;
        }
        if (key < temp->key)
        {
            suc = temp;
            temp = temp->left;
        }
        else
        {
            pre = temp;
            temp = temp->right;
        }
    }
}

// Floor Value Node: Node with the greatest data lesser than or equal to the key value. 
// Ceil Value Node: Node with the smallest data larger than or equal to the key value.
void floorCeilBST(Node *root, int key, int &floor, int &ceil)
{

    while (root)
    {

        if (root->data == key)
        {
            ceil = root->data;
            floor = root->data;
            return;
        }

        if (key > root->data)
        {
            floor = root->data;
            root = root->right;
        }
        else
        {
            ceil = root->data;
            root = root->left;
        }
    }
    return;
}

bool solve(Node *root, int &ans, int &mn, int &mx, int &size)
{
    if (root == NULL)
    {
        return true;
    }
    int mn1 = 10000000;
    int mx1 = -1;
    int size1 = 0;

    bool f1 = solve(root->left, ans, mn1, mx1, size1);

    int mn2 = 10000000;
    int mx2 = -1;
    int size2 = 0;

    bool f2 = solve(root->right, ans, mn2, mx2, size2);

    if (f1 && f2 && mx1 < root->data && mn2 > root->data)
    {
        ans = max(ans, size1 + size2 + 1);
        mn = min(mn1, min(root->data, mn2));
        mx = max(mx1, max(root->data, mx2));
        size += size1 + size2 + 1;
        return true;
    }
    mn = min(mn1, min(root->data, mn2));
    mx = max(mx1, max(root->data, mx2));

    return false;
}

/*You are required to complete this method */
// Return the size of the largest sub-tree which is also a BST
int largestBst(Node *root)
{
    //Your code here
    int ans = 0;
    int mn = 10000000;
    int mx = -1;
    int size = 0;
    solve(root, ans, mn, mx, size);

    return ans;
}


string serialize(TreeNode *root)
{
    string ans = "";
    queue<TreeNode *> q;
    q.push(root);
    while (!q.empty())
    {
        TreeNode *x = q.front();
        q.pop();
        if (x == NULL)
        {
            ans += "N,";
            continue;
        }
        else
        {
            ans += to_string(x->val) + ',';
        }
        if (x->left)
        {
            q.push(x->left);
        }
        else
        {
            q.push(NULL);
        }
        if (x->right)
        {
            q.push(x->right);
        }
        else
        {
            q.push(NULL);
        }
    }
    return ans;
}

// Decodes your encoded data to tree.

TreeNode *deserialize(string data)
{
    queue<TreeNode *> q1;
    int n = data.length();
    for (int i = 0; i < n;)
    {
        string x = "";
        while (i < n && data[i] != ',')
        {
            x += data[i];
            i++;
        }
        if (x == "N")
        {
            q1.push(NULL);
        }
        else
        {
            q1.push(new TreeNode(stoi(x)));
        }
        i++;
    }
    TreeNode *root = q1.front();
    queue<TreeNode *> q2;
    q2.push(q1.front());
    q1.pop();
    while (!q1.empty())
    {
        TreeNode *cur = q2.front();
        q2.pop();
        cur->left = q1.front();
        if (q1.front())
            q2.push(q1.front());
        q1.pop();

        if (q1.empty())
        {
            break;
        }

        cur->right = q1.front();
        if (q1.front())
            q2.push(q1.front());
        q1.pop();
    }
    return root;
}
