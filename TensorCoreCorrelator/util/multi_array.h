#if !defined MULTI_ARRAY_H
#define MULTI_ARRAY_H

// Experimental class that provides multi-dimensional arrays, similar to
// boost::multi_array[_ref], but much simpler.

namespace multi_array {
  template <std::size_t dim> class extent {
    public:
      extent(std::size_t size, const extent<dim - 1> &rhs) : size(rhs.size * size), next_extent(extent<dim - 1>(size, rhs.next_extent)) {}
      extent<dim + 1> operator [] (std::size_t size) const { return extent<dim + 1>(size, *this); }

      const std::size_t size;
      const extent<dim - 1> next_extent;
  };


  template <> class extent<0> {
    public:
      extent() {}
      template <typename T> extent(std::size_t, T) {}
      extent<1> operator [] (std::size_t size) const { return extent<1>(size, *this); }

      static const std::size_t size = 1;
      static const struct {} next_extent;
  };


  const static extent<0> extents;


  template<typename T, std::size_t dim> class array_ref {
    public:
      array_ref(T &ref, const extent<dim> &extents) : ref(ref), extents(extents) {}
      template <std::size_t d = dim> typename std::enable_if<d == 1, T &>::type operator [] (std::size_t index) const { return (&ref)[index]; }
      template <std::size_t d = dim> typename std::enable_if<d != 1, array_ref<T, dim - 1>>::type operator [] (std::size_t index) const { return array_ref<T, dim - 1>((&ref)[index * extents.next_extent.size], extents.next_extent); }
      T *begin() const { return &ref; }
      T *end() const { return &ref + extents.size; }
      std::size_t bytesize() const { return sizeof(T) * extents.size; }

      const extent<dim> extents;

    private:
      T &ref;
  };


  namespace detail {
    template <typename T> class array_base {
      protected:
	array_base(std::size_t size) : vec(size) {}
	std::vector<T> vec;
    };
  }


  template<typename T, std::size_t dim> class array : private detail::array_base<T>, public array_ref<T, dim> {
    public:
      array(const extent<dim> &extents) : detail::array_base<T>(extents.size), array_ref<T, dim>(*detail::array_base<T>::vec.data(), extents) {}
  };
}

#endif
