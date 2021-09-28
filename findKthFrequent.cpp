
int partition(vector<int>&count,int l,int r){
        int pivot=count[r];
        int i=l-1;
        int j=l;
        while(j<r){
            if(count[j]<pivot){
                int temp=count[j];
                count[j]=count[i+1];
                count[i+1]=temp;
                i++;
            }
            j++;
        }
        count[r]=count[i+1];
        count[i+1]=pivot;
        return i+1;
    }
    
    int quickSelect(vector<int>&count,int k,int l,int r){
        int part=partition(count,l,r);
        if(r-part+1==k){
            return count[part];
        }
        if(k>r-part+1)
        return quickSelect(count,k-(r-part+1),l,part-1);
        
        return quickSelect(count,k,part+1,r);
    }
    
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int,int>mp;
        for(auto it:nums){
            mp[it]++;
        }
        vector<int>count;
        for(auto it:mp){
            count.push_back(it.second);
        }
        int x=quickSelect(count,k,0,count.size()-1);
        vector<int>ans;
        for(auto it:mp){
            if(it.second>=x){
                ans.push_back(it.first);
            }
        }
        return ans;
    }