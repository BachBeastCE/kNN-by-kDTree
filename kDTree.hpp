#include "main.hpp"
#include "Dataset.hpp"
/* TODO: Please design your data structure carefully so that you can work with the given dataset
 *       in this assignment. The below structures are just some suggestions.
 */
struct kDTreeNode
{
    vector<int> data;
    kDTreeNode *left;
    kDTreeNode *right;
    kDTreeNode(vector<int> data, kDTreeNode *left = nullptr, kDTreeNode *right = nullptr)
    {
        this->data = data;
        this->left = left;
        this->right = right;
    }
    friend ostream &operator<<(ostream &os, const kDTreeNode &node)
    {
        os << "(";
        for (int i = 0; i < node.data.size(); i++)
        {
            os << node.data[i];
            if (i != node.data.size() - 1)
            {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }
};


class kDTree
{
private:
    int k;
    kDTreeNode *root;

    void inOrderRec(kDTreeNode* root) const;
    void preOrderRec(kDTreeNode* root) const;
    void postOrderRec(kDTreeNode* root) const;
    
    int heightRec(kDTreeNode* root) const;
    int nodeCountRec(kDTreeNode* root) const;
    int leafCountRec(kDTreeNode* node) const;

    kDTreeNode* insertRec(kDTreeNode *root, const vector<int> &point , int depth);
    bool searchRec(kDTreeNode* root, const vector<int>& point, int depth);

    kDTreeNode* min_Of_3_kDTreeNode(kDTreeNode* x, kDTreeNode* y, kDTreeNode* z, int dimension);
    kDTreeNode* findMinRec(kDTreeNode* root, int dimension, int depth);
    kDTreeNode* findMin(kDTreeNode* root, int dimension);
    kDTreeNode* removeRec(kDTreeNode*& root, const vector<int>& point, int depth);
    kDTreeNode* buildTreeRec(const vector<vector<int>> &points, int depth);

public:
    kDTree(int k = 2);
    ~kDTree();

    void clearRec(kDTreeNode* root);
    kDTreeNode* copyTree(kDTreeNode* root);
    const kDTree &operator=(const kDTree &other);
    kDTree(const kDTree &other);

    void inorderTraversal() const;
    void preorderTraversal() const;
    void postorderTraversal() const;
    int height() const;
    int nodeCount() const;
    int leafCount() const;

    void insert(const vector<int> &point);
    void remove(const vector<int> &point);
    bool search(const vector<int> &point);

    vector<vector<int>> merge (vector<vector<int>> &left, vector<vector<int>> &right, int dim);
    vector<vector<int>> mergeSort( vector<vector<int>> &points, int dim);
    
    void buildTree(const vector<vector<int>> &pointList);

    kDTreeNode* closestpoint(kDTreeNode* node1, kDTreeNode* node2, const vector<int>& target);
    double distSquared(const vector<int>& point1, const vector<int>& point2);
    kDTreeNode* nearestNeighbourRec(kDTreeNode* root, const vector<int>& target, int depth); 
    void nearestNeighbour(const vector<int> &target, kDTreeNode *&best);

    void kNearestNeighbourRec(kDTreeNode* temp, const vector<int>& target, int k, vector<kDTreeNode*>& bestList, int level);
    void kNearestNeighbour(const vector<int> &target, int k, vector<kDTreeNode *> &bestList);

};

class kNN
{
private:
    int k;
    kDTree* tree;
    Dataset* X_train;
    Dataset* Y_train;

public:
    kNN(int k = 5);
    void fit(Dataset &X_train, Dataset &y_train);
    Dataset predict(Dataset &X_test);
    double score(const Dataset &y_test, const Dataset &y_pred); 
};

// Please add more or modify as needed




