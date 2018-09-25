#include "mpi.h"
#include <algorithm>
#include <fstream>
#include <vector>

using namespace std;

const int MAX_WORD_LENGTH = 50;

struct Word {
  char word[MAX_WORD_LENGTH];
};

struct Pair {
  char word[MAX_WORD_LENGTH];
  int count;
  Pair() {
    this->count = 1;
  }
};

// comparator for sorting pairs in slaves
bool compare_pairs(const Pair &p1, const Pair &p2) {
  return strcmp(p1.word, p2.word) < 0;
}

// sends the given vector to the given destination
// sends the vector as a char array
template <typename T>
void send(const int &dest, const vector<T> &sent) {
  MPI_Send((void*)sent.data(), sent.size() * sizeof(T), MPI_BYTE, dest, 1, MPI_COMM_WORLD);
}

// receives a vector of the given type from the given source
// receives the vector as a char array
template <typename T>
vector<T> receive(const int &source, const T &type) {
  int receive_size;
  MPI_Status status;
  MPI_Probe(source, 1, MPI_COMM_WORLD, &status);
  MPI_Get_count(&status, MPI_BYTE, &receive_size);
  vector<T> received(receive_size / sizeof(T));
  MPI_Recv((void*)received.data(), receive_size, MPI_BYTE, source, 1, MPI_COMM_WORLD, &status);
  return received;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank; // rank of the current processor
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // gets the rank of the current processor
  int size; // total number of processors
  MPI_Comm_size(MPI_COMM_WORLD, &size); // gets the total number of processors
  if (rank == 0) {
    // default input and output file names
    string input_file = "speech_tokenized.txt", output_file = "reduced.txt";
    if (argc > 1) {
      input_file = argv[1];   // input file name is taken from arguments
      output_file = argv[2];  // output file name is taken from arguments
    }
    // master reads words
    FILE *file = fopen(input_file.c_str(), "r");
    char word[MAX_WORD_LENGTH];
    vector<Word> all_words;
    while (fscanf(file, "%s", word) != EOF) {
      Word w;
      strcpy(w.word, word);
      all_words.push_back(w);
    }
    fclose(file);
    int elements_per_cpu = all_words.size() / (size - 1);
    // master splits the words and sends them to slaves
    for (int i = 1; i < size; i++) {
      send(i, vector<Word>(&all_words[(i - 1) * elements_per_cpu], &all_words[i != size - 1 ? i * elements_per_cpu : all_words.size()]));
    }
    vector<Pair> all_pairs;
    // master receives pairs
    for (int i = 1; i < size; i++) {
      vector<Pair> received_pairs = receive(i, Pair());
      all_pairs.insert(all_pairs.end(), received_pairs.begin(), received_pairs.end());
    }
    // master splits the pairs and sends them to slaves
    for (int i = 1; i < size; i++) {
      send(i, vector<Pair>(&all_pairs[(i - 1) * elements_per_cpu], &all_pairs[i != size - 1 ? i * elements_per_cpu : all_pairs.size()]));
    }
    all_pairs.clear();
    // master receives sorted pairs and does last merge
    for (int i = 1; i < size; i++) {
      for (Pair p : receive(i, Pair())) {
        bool inserted = false;
        for (vector<Pair>::iterator it = all_pairs.begin(); it != all_pairs.end(); it++) {
          if (strcmp(p.word, it->word) < 0) {
            all_pairs.insert(it, p);
            inserted = true;
            break;
          }
        }
        if (!inserted) {
          all_pairs.push_back(p);
        }
      }
    }
    // master reduces pairs
    vector<Pair>::iterator prev = all_pairs.begin();
    for (vector<Pair>::iterator it = all_pairs.begin() + 1; it != all_pairs.end();) {
      if (strcmp(prev->word, it->word) == 0) {
        it = all_pairs.erase(it);
        prev->count++;
      } else {
        prev = it++;
      }
    }
    // master prints reduced pairs
    file = fopen(output_file.c_str(), "w");
    for (Pair p : all_pairs) {
      fprintf(file, "%s %d\n", p.word, p.count);
    }
    fclose(file);
  } else {
    vector<Pair> all_pairs;
    // slaves receive words, create pairs from words and send them to master
    for (Word w : receive(0, Word())) {
      Pair p;
      strcpy(p.word, w.word);
      all_pairs.push_back(p);
    }
    send(0, all_pairs);
    // slaves receive pairs, sort them and send them to master
    all_pairs = receive(0, Pair());
    sort(all_pairs.begin(), all_pairs.end(), compare_pairs);
    send(0, all_pairs);
  }
  MPI_Finalize();
  return 0;
}
