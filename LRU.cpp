#include <bits/stdc++.h>
using namespace std;
struct Node
{
    int data;
    int key;
    Node *prev;
    Node *next;
    Node()
    {
        prev = NULL;
        next = NULL;
    }
    Node(int d, int k)
    {
        data = d;
        key = k;
        prev = NULL;
        next = NULL;
    }
};

class LRUCache
{
    unordered_map<int, Node *> mp;
    Node *head;
    Node *tail;
    int current_size;
    int total_size;

public:
    LRUCache(int capacity)
    {
        current_size = 0;
        total_size = capacity;
        head = NULL;
        tail = NULL;
    }

    Node *push_front(Node *temp)
    {
        if (head == NULL)
        {
            head = temp;
            tail = temp;
            return head;
        }

        temp->next = head;
        head->prev = temp;
        return temp;
    }

    int pop_back()
    {
        int key = tail->key;
        if (head == tail)
        {
            Node *temp = head;
            head = NULL;
            tail = NULL;
            delete temp;
            return key;
        }

        Node *temp = tail;
        tail = temp->prev;
        tail->next = NULL;
        delete temp;
        return key;
    }

    void removeFromMiddle(Node *temp)
    {
        if (head == tail)
        {
            head = NULL;
            tail = NULL;
            return;
        }

        if (temp == head)
        {
            head = head->next;
            head->prev = NULL;
            return;
        }

        Node *p = temp->prev;
        Node *n = temp->next;
        p->next = n;
        if (n)
        {
            n->prev = p;
        }
        else
        {
            tail = p;
        }
    }

    int get(int key)
    {
        if (mp.find(key) != mp.end())
        {
            Node *curr = mp[key];
            removeFromMiddle(curr);
            head = push_front(curr);
            return head->data;
        }
        return -1;
    }

    void put(int key, int value)
    {
        if (mp.find(key) != mp.end())
        {
            Node *curr = mp[key];
            curr->data = value;

            removeFromMiddle(curr);
            head = push_front(curr);
            return;
        }
        if (current_size < total_size)
        {
            head = push_front(new Node(value, key));
            mp.insert({key, head});
            current_size++;
        }
        else
        {

            mp.erase(pop_back());
            head = push_front(new Node(value, key));
            mp.insert({key, head});
        }
    }
};