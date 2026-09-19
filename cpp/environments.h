#include <cstdint>
#include <vector>
#include <fstream>
#include <array>

/*** Abstract Environment Class ***/
class Environment {
  public:
		virtual ~Environment()=0;
    virtual Environment *getNextState(int action) const = 0;

    virtual std::vector<Environment*> getNextStates() const = 0;

    virtual std::vector<uint8_t> getState() const = 0;

    virtual bool isSolved() const = 0;

    virtual int getNumActions() const = 0;
};

/*** PuzzleN ***/
class PuzzleN: public Environment {
	private:
		static const int numActions = 4;
		uint8_t **swapZeroIdxs;

		std::vector<uint8_t> state;
		uint8_t dim;
		int numTiles;
		uint8_t zIdx;

    virtual void construct(std::vector<uint8_t> state, uint8_t dim, uint8_t zIdx);
	public:
		PuzzleN(std::vector<uint8_t> state, uint8_t dim, uint8_t zIdx);
		PuzzleN(std::vector<uint8_t> state, uint8_t dim);
		~PuzzleN();

		
    virtual PuzzleN *getNextState(int action) const;

    virtual std::vector<Environment*> getNextStates() const;

    virtual std::vector<uint8_t> getState() const;

    virtual bool isSolved() const;

    virtual int getNumActions() const;
};

/*** LightsOut ***/
class LightsOut: public Environment {
	private:
		int **moveMat;

		std::vector<uint8_t> state;
		uint8_t dim;
		int numActions;
	public:
		LightsOut(std::vector<uint8_t> state, uint8_t dim);
		~LightsOut();

		
    virtual LightsOut *getNextState(int action) const;

    virtual std::vector<Environment*> getNextStates() const;

    virtual std::vector<uint8_t> getState() const;

    virtual bool isSolved() const;

    virtual int getNumActions() const;
};

class Cube3: public Environment {
	private:
		static const int numActions = 12;
		static constexpr int rotateIdxs_old[12][24] = 
		{
			{2, 5, 8, 8, 7, 6, 6, 3, 0, 0, 1, 2, 38, 41, 44, 20, 23, 26, 47, 50, 53, 29, 32, 35},
			{6, 3, 0, 0, 1, 2, 2, 5, 8, 8, 7, 6, 47, 50, 53, 29, 32, 35, 38, 41, 44, 20, 23, 26},
			{11, 14, 17, 17, 16, 15, 15, 12, 9, 9, 10, 11, 45, 48, 51, 18, 21, 24, 36, 39, 42, 27, 30, 33},
			{15, 12, 9, 9, 10, 11, 11, 14, 17, 17, 16, 15, 36, 39, 42, 27, 30, 33, 45, 48, 51, 18, 21, 24},
			{20, 23, 26, 26, 25, 24, 24, 21, 18, 18, 19, 20, 45, 46, 47, 0, 1, 2, 44, 43, 42, 9, 10, 11},
			{24, 21, 18, 18, 19, 20, 20, 23, 26, 26, 25, 24, 44, 43, 42, 9, 10, 11, 45, 46, 47, 0, 1, 2},
			{29, 32, 35, 35, 34, 33, 33, 30, 27, 27, 28, 29, 38, 37, 36, 6, 7, 8, 51, 52, 53, 15, 16, 17},
			{33, 30, 27, 27, 28, 29, 29, 32, 35, 35, 34, 33, 51, 52, 53, 15, 16, 17, 38, 37, 36, 6, 7, 8},
			{38, 41, 44, 44, 43, 42, 42, 39, 36, 36, 37, 38, 18, 19, 20, 2, 5, 8, 35, 34, 33, 15, 12, 9},
			{42, 39, 36, 36, 37, 38, 38, 41, 44, 44, 43, 42, 35, 34, 33, 15, 12, 9, 18, 19, 20, 2, 5, 8},
			{47, 50, 53, 53, 52, 51, 51, 48, 45, 45, 46, 47, 29, 28, 27, 0, 3, 6, 24, 25, 26, 17, 14, 11},
			{51, 48, 45, 45, 46, 47, 47, 50, 53, 53, 52, 51, 24, 25, 26, 17, 14, 11, 29, 28, 27, 0, 3, 6}
		};

		static constexpr int rotateIdxs_new[12][24] = 
		{
			{0, 1, 2, 2, 5, 8, 8, 7, 6, 6, 3, 0, 20, 23, 26, 47, 50, 53, 29, 32, 35, 38, 41, 44},
			{0, 1, 2, 2, 5, 8, 8, 7, 6, 6, 3, 0, 20, 23, 26, 47, 50, 53, 29, 32, 35, 38, 41, 44},
			{9, 10, 11, 11, 14, 17, 17, 16, 15, 15, 12, 9, 18, 21, 24, 36, 39, 42, 27, 30, 33, 45, 48, 51},
			{9, 10, 11, 11, 14, 17, 17, 16, 15, 15, 12, 9, 18, 21, 24, 36, 39, 42, 27, 30, 33, 45, 48, 51},
			{18, 19, 20, 20, 23, 26, 26, 25, 24, 24, 21, 18, 0, 1, 2, 44, 43, 42, 9, 10, 11, 45, 46, 47},
			{18, 19, 20, 20, 23, 26, 26, 25, 24, 24, 21, 18, 0, 1, 2, 44, 43, 42, 9, 10, 11, 45, 46, 47},
			{27, 28, 29, 29, 32, 35, 35, 34, 33, 33, 30, 27, 6, 7, 8, 51, 52, 53, 15, 16, 17, 38, 37, 36},
			{27, 28, 29, 29, 32, 35, 35, 34, 33, 33, 30, 27, 6, 7, 8, 51, 52, 53, 15, 16, 17, 38, 37, 36},
			{36, 37, 38, 38, 41, 44, 44, 43, 42, 42, 39, 36, 2, 5, 8, 35, 34, 33, 15, 12, 9, 18, 19, 20},
			{36, 37, 38, 38, 41, 44, 44, 43, 42, 42, 39, 36, 2, 5, 8, 35, 34, 33, 15, 12, 9, 18, 19, 20},
			{45, 46, 47, 47, 50, 53, 53, 52, 51, 51, 48, 45, 0, 3, 6, 24, 25, 26, 17, 14, 11, 29, 28, 27},
			{45, 46, 47, 47, 50, 53, 53, 52, 51, 51, 48, 45, 0, 3, 6, 24, 25, 26, 17, 14, 11, 29, 28, 27}
		};

		std::vector<uint8_t> state;
	public:
		Cube3(std::vector<uint8_t> state);
		~Cube3();

		
    virtual Cube3 *getNextState(int action) const;

    virtual std::vector<Environment*> getNextStates() const;

    virtual std::vector<uint8_t> getState() const;

    virtual bool isSolved() const;

    virtual int getNumActions() const;
};

class Puppet: public Environment {
	private:
		static const int numActions = 9;

		static constexpr uint8_t numCorners = 7;
    	static constexpr uint8_t numEdges = 9;
		static constexpr uint8_t numCornerVals = 21;
		static constexpr uint8_t numEdgeVals = 18;

		static constexpr int rotate_corners[numActions][numCornerVals] = {
    			{17, 14, 8, 9, 4, 5, 6, 3, 0, 15, 16, 11, 12, 13, 10, 7, 1, 2, 18, 19, 20}, // R
    			{8, 16, 17, 7, 4, 5, 6, 15, 2, 3, 14, 11, 12, 13, 1, 9, 10, 0, 18, 19, 20}, // R'
    			{2, 10, 0, 15, 4, 5, 6, 9, 17, 7, 1, 11, 12, 13, 16, 3, 14, 8, 18, 19, 20}, // R2
    			{12, 1, 2, 0, 10, 11, 6, 19, 8, 9, 7, 17, 18, 13, 5, 15, 16, 14, 3, 4, 20}, // U
    			{3, 1, 2, 18, 19, 14, 6, 10, 8, 9, 4, 5, 0, 13, 17, 15, 16, 11, 12, 7, 20}, // U'
    			{18, 1, 2, 12, 7, 17, 6, 4, 8, 9, 19, 14, 3, 13, 11, 15, 16, 5, 0, 10, 20}, // U2
    			{1, 13, 2, 3, 4, 7, 12, 8, 20, 9, 10, 11, 14, 19, 15, 6, 16, 17, 18, 0, 5}, // F
    			{19, 0, 2, 3, 4, 20, 15, 5, 7, 9, 10, 11, 6, 1, 12, 14, 16, 17, 18, 13, 8}, // F'
    			{13, 19, 2, 3, 4, 8, 14, 20, 5, 9, 10, 11, 15, 0, 6, 12, 16, 17, 18, 1, 7}  // F2
		};

		static constexpr int rotate_edges[numActions][numEdgeVals] = {
			{0, 11, 7, 3, 4, 5, 1, 15, 8, 9, 2, 16, 12, 13, 14, 10, 6, 17}, // R
			{0, 6, 10, 3, 4, 5, 16, 2, 8, 9, 15, 1, 12, 13, 14, 7, 11, 17}, // R'
		    	{0, 16, 15, 3, 4, 5, 11, 10, 8, 9, 7, 6, 12, 13, 14, 2, 1, 17}, // R2
		    	{3, 1, 9, 17, 4, 5, 6, 7, 2, 12, 10, 0, 8, 13, 14, 15, 16, 11}, // U
		    	{11, 1, 8, 0, 4, 5, 6, 7, 12, 2, 10, 17, 9, 13, 14, 15, 16, 3}, // U'
		    	{17, 1, 12, 11, 4, 5, 6, 7, 9, 8, 10, 3, 2, 13, 14, 15, 16, 0}, // U2
		    	{10, 5, 2, 3, 0, 13, 6, 7, 8, 1, 14, 11, 12, 9, 4, 15, 16, 17}, // F
		    	{4, 9, 2, 3, 14, 1, 6, 7, 8, 13, 0, 11, 12, 5, 10, 15, 16, 17}, // F'
		    	{14, 13, 2, 3, 10, 9, 6, 7, 8, 5, 4, 11, 12, 1, 0, 15, 16, 17}  // F2
		};

		std::vector<uint8_t> state;

		static constexpr std::array<uint8_t, numCornerVals> position =
		{0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6};
    	static constexpr std::array<uint8_t, numCornerVals> orientation =
		{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2};
		static constexpr std::array<uint16_t, 7> pow3 = {0, 1, 3, 9, 27, 81, 243};

		inline static const auto from7choose3 = [] {
			std::array<std::array<uint8_t, 128>, 7> table{};
			for (auto& row : table) {
				row.fill(255);
			}
			uint8_t idx = 0;
			for(int i = 0; i < numCorners; i++){
				for(int j = 0; j < numCorners; j++){
					for(int k = 0; k < numCorners; k++){
						for(int zeroCor = 0; zeroCor < numCorners; zeroCor++){
							if(zeroCor != i && zeroCor != j && zeroCor != k && i < j && j < k){
								uint8_t permutation = 0;
								permutation |= (1u << i);
								permutation |= (1u << j);
								permutation |= (1u << k);
								table[zeroCor][permutation] = idx++;
							}
						}
					}
				}
			}
			return table;
		}();

	public:
		Puppet(std::vector<uint8_t> state);
		~Puppet();

		
    virtual Puppet *getNextState(int action) const;

    virtual std::vector<Environment*> getNextStates() const;

    virtual std::vector<uint8_t> getState() const;

    virtual bool isSolved() const;

    virtual int getNumActions() const;
};

class Cube4: public Environment {
	private:
		static const int numActions = 24;
		static const std::vector<std::vector<int> > rotateIdxs_old;
		static const std::vector<std::vector<int> > rotateIdxs_new;

		std::vector<uint8_t> state;
	public:
		Cube4(std::vector<uint8_t> state);
		~Cube4();
		
    virtual Cube4 *getNextState(int action) const;

    virtual std::vector<Environment*> getNextStates() const;

    virtual std::vector<uint8_t> getState() const;

    virtual bool isSolved() const;

    virtual int getNumActions() const;
};


