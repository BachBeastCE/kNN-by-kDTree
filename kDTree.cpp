#include "kDTree.hpp"

    kDTree::kDTree(int k)
    {
        this->k = k;
        root = nullptr;
    }
    void kDTree::clearRec(kDTreeNode* root)
    {
        if (root != nullptr){
            clearRec(root->left);
            clearRec(root->right);
            delete(root);
            if (root->left != nullptr) root->left = nullptr;
            if (root->right != nullptr) root->right = nullptr;
            root = nullptr;
        }
        else return;
    }

    kDTree::~kDTree()
    {   
        clearRec (root);
        root = nullptr;
    }

    kDTreeNode* kDTree::copyTree(kDTreeNode* root)
    {
        if (root == nullptr)return nullptr;
        kDTreeNode* newRoot = new kDTreeNode(root->data);
        newRoot->left = copyTree(root->left);
        newRoot->right = copyTree(root->right);
        return newRoot;
    }

    const kDTree & kDTree::operator=(const kDTree &other)
    {
    if (this == &other) return *this;
    this->k = other.k;
    if (other.root == nullptr) {
        this->root = nullptr;
    } else {
        this->root = copyTree(other.root);
    }
    return *this;
    }

    kDTree::kDTree(const kDTree &other)
    {
    this->k = other.k;
    if (other.root == nullptr) {
        this->root = nullptr;
    } else {
        this->root = copyTree(other.root);
    }
    }

//////////////////////////////////////////////////////////////////////////

    void kDTree::preOrderRec(kDTreeNode* root) const {
    if (root == nullptr) return;
    cout << "(";
    for (size_t i = 0; i < root->data.size(); i++){
        cout << root->data[i];
        if (i == root->data.size() - 1){
            cout << ")";
        }else{
            cout << ", ";
        }
    }
    cout <<" ";

    preOrderRec(root->left);
    preOrderRec(root->right);
    }

    void kDTree::inOrderRec(kDTreeNode* root) const
    {
    if (root == nullptr) return;
    
    inOrderRec(root->left);

    cout << "(";
    for (size_t i = 0; i < root->data.size(); i++){
        cout << root->data[i];
        if (i == root->data.size() - 1){
            cout << ")";
        }else{
            cout << ", ";
        }
    }
    cout <<" ";

    inOrderRec(root->right);
    }

    void kDTree::postOrderRec(kDTreeNode* root) const
    {
    if (root == nullptr) return;
    
    postOrderRec(root->left);
    postOrderRec(root->right);
    cout << "(";
    for (size_t i = 0; i < root->data.size(); i++){
        cout << root->data[i];
        if (i == root->data.size() - 1){
            cout << ")";
        }else{
            cout << ", ";
        }
    }
    cout <<" ";
    }

    void kDTree::preorderTraversal() const
    {
        preOrderRec(this->root);
    }

    void kDTree::inorderTraversal() const
    {
        inOrderRec(this->root);
    }

    void kDTree::postorderTraversal() const
    {
        postOrderRec(this->root);
    }

    //////////////////////////////////////////////////////////////////////////

int kDTree::heightRec(kDTreeNode* root) const
{
    if (!root) return 0;
    int leftHeight = heightRec(root->left);
    int rightHeight = heightRec(root->right);
    return std::max(leftHeight, rightHeight) + 1;
}

int kDTree::height() const
    {
        return heightRec(this->root);
    }
    
int kDTree::nodeCountRec(kDTreeNode* root) const {
        if (root == nullptr) return 0;
        return 1 + nodeCountRec(root->left) + nodeCountRec(root->right);
    }

int kDTree::nodeCount() const
    {
        return nodeCountRec(this->root);
    }
   
int kDTree::leafCountRec(kDTreeNode* node) const
    {
        if (node == nullptr) return 0;
        if (node->left == nullptr && node->right == nullptr) return 1;
        else return leafCountRec(node->left) + leafCountRec(node->right);
    }        

int kDTree::leafCount() const
    {
        return leafCountRec(this->root);
    }

kDTreeNode* kDTree::insertRec(kDTreeNode *root, const vector<int> &point , int depth)
    {
    if (root == nullptr)
    {   
       kDTreeNode* newnode = new kDTreeNode(point);
       return newnode;
    }
 
    int cd = depth % k;
 
    if (point[cd] < (root->data[cd]))
        root->left  = insertRec(root->left, point, depth + 1);
    else
        root->right = insertRec(root->right, point, depth + 1);
 
    return root;
    }
    
void kDTree::insert(const vector<int> &point)
    {
        if (point.size() != k)return;
        this->root = insertRec(this->root, point, 0);
    }

bool kDTree::searchRec(kDTreeNode* root, const vector<int>& point, int depth) 
    {
    if (root == nullptr) return false;
    unsigned cd = depth % k;
    if (root->data==point) return true;
    if (point[cd] < root->data[cd]) return searchRec(root->left, point, depth + 1);
    return searchRec(root->right, point, depth + 1);
    }

bool kDTree::search(const vector<int>& point) {
    return searchRec(root,point,0);
    }

kDTreeNode* kDTree::min_Of_3_kDTreeNode(kDTreeNode* x, kDTreeNode* y, kDTreeNode* z, int dimension)
    {
        kDTreeNode* result = x;
        if (y != nullptr && y->data[dimension] < result->data[dimension])
            result = y;
        if (z != nullptr && z->data[dimension] < result->data[dimension])
            result = z;
        return result;
    }

kDTreeNode* kDTree::findMinRec(kDTreeNode* root, int dimension, int depth) {
    if (root == nullptr) return nullptr;
    unsigned current_dimension = depth % k;
    if (current_dimension == dimension) {
    if (root->left == nullptr) return root;
        return findMinRec(root->left, dimension, depth + 1);
    }
    return min_Of_3_kDTreeNode(root,
        findMinRec(root->left, dimension, depth + 1),
        findMinRec(root->right, dimension, depth + 1), dimension);
}

kDTreeNode* kDTree::findMin(kDTreeNode* root, int dimension) {
    return findMinRec(root, dimension,0);
}

kDTreeNode* kDTree::removeRec(kDTreeNode*& root, const vector<int>& point, int depth) {
    if (root == nullptr) return nullptr;
    int cd = depth % k;
    if (root->data==point){
        if(root->right == nullptr && root->left == nullptr){
            delete root;
            return nullptr;
        }
        else if(root->right != nullptr){
            kDTreeNode* min = findMinRec(root->right,cd, depth+1);
            root->data=min->data;   
            root->right = removeRec(root->right, min->data, depth + 1);
        }
        else if( root->left != nullptr){//có cây con trái 
            kDTreeNode* min = findMinRec(root->left, cd , depth+1);
            root->data=min->data; 
            root->right= root->left;
            root->left = nullptr;
            root->right = removeRec (root->right,min->data,depth + 1);
        }
        return root;
    } 
    if (point[cd] < root->data[cd]) root->left = removeRec(root->left, point,depth+1);
    else root->right = removeRec(root->right, point, depth+1);
    return root;
}

void kDTree::remove(const vector<int> &point)
    {
        this->root = removeRec(this->root, point, 0);
    }

    /////////////////////////////////////////////////////////////////////

vector<vector<int>>kDTree::mergeSort( vector<vector<int>> &points, int dimension){
    if (points.size() == 1){
        return points;
    }

    int mid = points.size() / 2;
    vector<vector<int>> left (points.begin(), points.begin() + mid);
    vector<vector<int>> right (points.begin() + mid, points.end());
    left = mergeSort(left, dimension);
    right = mergeSort(right, dimension);

    return merge(left, right, dimension);
}

vector<vector<int>> kDTree::merge (vector<vector<int>> &left, vector<vector<int>> &right, int dimension){
    vector<vector<int>> result;
    int leftIndex = 0, rightIndex = 0;
    while (leftIndex < left.size() && rightIndex < right.size()){
        if (left[leftIndex][dimension] < right[rightIndex][dimension]){
            result.push_back(left[leftIndex]);
            leftIndex++;
        }else {
            result.push_back(right[rightIndex]);
            rightIndex++;
        }
    }

    while (leftIndex < left.size()){
        result.push_back(left[leftIndex]);
        leftIndex++;
    }

    while (rightIndex < right.size()){
        result.push_back(right[rightIndex]);
        rightIndex++;
    }
    return result;
}

kDTreeNode* kDTree::buildTreeRec(const vector<vector<int>> &points, int depth){
    if (points.empty()){
        return nullptr;
    }
    int k = points[0].size();
    int dim = depth % k;

    vector<vector<int>> sortedPoints = points;
    sortedPoints = mergeSort(sortedPoints,dim);

    size_t medianIndex = 0;
    if (sortedPoints.size() % 2 == 0){
        medianIndex = (sortedPoints.size() / 2) - 1;
    }else {
        medianIndex = sortedPoints.size() / 2;
    }

    vector<int> medianPoint = sortedPoints[medianIndex];

    kDTreeNode* newNode = new kDTreeNode(medianPoint);


    newNode->left = buildTreeRec(vector<vector<int>>(sortedPoints.begin(), sortedPoints.begin() + medianIndex), depth +1);
    newNode->right = buildTreeRec(vector<vector<int>>(sortedPoints.begin() + medianIndex + 1, sortedPoints.end()), depth +1);

    return newNode;
}

void kDTree::buildTree(const vector<vector<int>> &pointList)
{ 
    if (root != nullptr) clearRec(this->root);
    this->k = pointList[0].size();
    this->root = buildTreeRec(pointList, 0);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////


    double kDTree::distSquared(const vector<int>& point1, const vector<int>& point2) {
        double Euclide_distance = 0;
        for (int i = 0; i < point1.size(); i++)
        {
            Euclide_distance += pow(point1[i] - point2[i], 2);
        }
        return Euclide_distance;
    }
    
    kDTreeNode* kDTree::closestpoint(kDTreeNode* node1, kDTreeNode* node2, const vector<int>& target) 
    {
        if (node1 == nullptr) return node2;

        if (node2 == nullptr) return node1;

        double d1 = distSquared(node1->data, target);
        double d2 = distSquared(node2->data, target);

        if (d1 < d2)
           return node1;
        else
            return node2;
    }
    
    kDTreeNode* kDTree::nearestNeighbourRec(kDTreeNode* root, const vector<int>& target, int depth) 
    {
        if (root == nullptr) return nullptr; //Điều kiện dừng
        kDTreeNode* nextBranch = nullptr;
        kDTreeNode* otherBranch = nullptr;

        //Nếu chiều hiện tại của target nhỏ hơn thì nhánh tiếp theo là cây con bên trái
        if (target[depth%k] < root->data[depth%k]) {
            nextBranch = root->left;
            otherBranch = root->right;
        }
        //Ngược lại
        else {
            nextBranch = root->right;
            otherBranch = root->left;
        }   
        kDTreeNode* temp = nearestNeighbourRec(nextBranch, target, depth + 1); //Gọi đệ quy đến nhanh tiếp theo
        kDTreeNode* best = closestpoint(temp, root, target); // Tìm ra điểm gần trên các nhánh tiếp theo đã qua
        //Tính bình phương khoảng cách và khoảng cách từ root đến đường thẳng trục
        long radiusSquared = distSquared(target, best->data);
        long dist = target[depth] - root->data[depth];
        //Trong trường hợp radiusSquare lớn hơn dist^2 thì có khả năng điểm gần nhất nằm ở nhánh còn lại
        if (radiusSquared >= dist * dist) {
            temp = nearestNeighbourRec(otherBranch, target, depth + 1); //Gọi đệ quy đến nhanh còn lại
            best = closestpoint(temp, best, target); // Tìm ra điểm gần nhất
        }
        return best;
    }

    void kDTree::kDTree::nearestNeighbour(const vector<int>& target, kDTreeNode* &best)
    {
        best = nearestNeighbourRec(this->root, target, 0);
        vector<int> data = best->data;
        best = new kDTreeNode(data);
    }
    
    void kDTree::kNearestNeighbourRec(kDTreeNode* root, const vector<int>& target, int k, vector<kDTreeNode*>& bestList, int depth) {
        if (root == nullptr) return; //Điều kiện dừng
        int alpha = depth % this->k; // Xác định chiều hiện tại
        kDTreeNode* nextBranch = nullptr;
        kDTreeNode* otherBranch = nullptr;
        if (root->data[alpha] > target[alpha])
        {
            nextBranch = root->left;
            otherBranch = root->right;
        }
        else
        {
            nextBranch = root->right;
            otherBranch = root->left;
        }

        kNearestNeighbourRec(nextBranch, target, k, bestList, depth + 1);
        int distance = int(distSquared(root->data, target));

        //Sort bestList theo khoang cach den target;
        if (!bestList.empty())
        { 
            vector<double> DisofNode;
            for (kDTreeNode* x : bestList){
                DisofNode.push_back(distSquared(target,x->data));
            }
            int n = DisofNode.size();
            for (int i = 0; i < n - 1; i++) {
                for (int j = 0; j < n - i - 1; j++) {
                    if (DisofNode[j] > DisofNode[j + 1]) {
                        swap(DisofNode[j], DisofNode[j + 1]);
                        swap(bestList[j],bestList[j + 1]);
                    }
                }
            }
        }

        // Thêm temp vào bestList nếu bestList chưa đủ k phần tử
        if (bestList.size() < k) {
            bestList.push_back(root);
        } else {
            // Tìm khoảng cách lớn nhất trong bestList
            int maxDistance = -1;
            size_t maxDistanceIndex = -1;
            for (size_t i = 0; i < bestList.size(); ++i) {
                int d = distSquared(bestList[i]->data, target);
                if (d > maxDistance) {
                    maxDistance = d;
                    maxDistanceIndex = i;
                }
            }
            // Nếu khoảng cách từ temp đến target nhỏ hơn khoảng cách lớn nhất trong bestList, thay thế nút tương ứng
            if (distance < maxDistance) {
                bestList[maxDistanceIndex] = root;
            }
        }
        int radiusSquared = int(distSquared(bestList.front()->data, target));
        int dist = target[alpha] - root->data[alpha];
        if ((bestList.size() <k) || (radiusSquared >= ( dist*dist ))) 
            kNearestNeighbourRec(otherBranch, target, k, bestList, depth + 1);
        return;
    }

    void kDTree::kNearestNeighbour(const vector<int>& target, int k, vector<kDTreeNode*>& bestList){
        kNearestNeighbourRec(this-> root, target,  k,bestList, 0);
    }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

vector<vector<int>> convertListToVector(const list<list<int>>& input){
    vector<vector<int>>  output;
    output.reserve(input.size());

    for (const auto& sublist : input){
        vector<int> element(sublist.begin(), sublist.end());
         output.push_back(element);
    }
    return  output;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////

    kNN::kNN(int k)
    {
        this->k = k;
    }
   
    void kNN::fit(Dataset &X_train, Dataset &y_train)
    {
    int x_row = 0, x_col = 0, y_row = 0, y_col = 0;
    X_train.getShape(x_row,x_col);
    y_train.getShape(y_row,y_col);
    X_train.data;
    int nRows, nCols;
    this->X_train = new Dataset(X_train);       //gan cho x_train
    this->Y_train = new Dataset(y_train);       //cho gan cho y_train
    this->X_train->getShape(nRows, nCols);
    this->tree = new kDTree(nCols);
    vector<vector<int>> pointList = convertListToVector(this->X_train->data);
    this->tree->buildTree(pointList);   
    }

    Dataset kNN::predict(Dataset& X_test) 
    {
        Dataset y_predict;
        y_predict.columnName = this->Y_train->columnName;
        for (auto& row : X_test.data) {
            vector<int> point;
            for (auto& data : row) {
                point.push_back(data);
            }
            vector<kDTreeNode *> bestList;
            this->tree->kNearestNeighbour(point, this->k, bestList);
            vector<int> PredLabel;
            for (auto& best : bestList) {
                int number = -1;
                auto data = this->X_train->data.begin();
                auto label = this->Y_train->data.begin();
                while(data != this->X_train->data.end() && label != this->Y_train->data.end()) {
                    if ((*data) == list<int>(best->data.begin(), best->data.end())){
                        number = (*label).front();
                        break;
                    }
                    data++;
                    label++;
                }
                PredLabel.push_back(number);
            }

            vector<int> count(10, 0);
            for (auto& label : PredLabel) {
                count[label]++;
            }
            int maxCount = 0, maxIndex = 0;
            for (int label = 0; label < 10; label++) {
                if (count[label] > maxCount) {
                    maxCount = count[label];
                    maxIndex = label;
                }
            }
            y_predict.data.push_back({maxIndex});
        }
        return y_predict;
    }
    
    double kNN::score(const Dataset& y_test, const Dataset& y_pred) 
    {
    double test = 0;
    double correct = 0;
    vector<int> y_test_arr;
    vector<int> y_pred_arr;
    for (const auto& innerList : y_pred.data) {
        std::vector<int> innerVector(innerList.begin(), innerList.end());
        y_pred_arr.push_back(innerVector[0]);
    }
    for (const auto& innerList : y_test.data) {
        std::vector<int> innerVector(innerList.begin(), innerList.end());
        y_test_arr.push_back(innerVector[0]);
    }
    for (int i = 0; i < y_test_arr.size(); i++)
    {
        if(y_test_arr[i]==y_pred_arr[i])correct+=1;
        test+=1;
    }
    return correct / test;
}
