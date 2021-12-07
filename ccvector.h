/*
 * CCLib
 *
 * collection of utils i use in most projects.
 * maybe it will evolve in a framework, maybe not
 *
 * (c) 2018 Carlo Casta <carlo.casta at gmail.com>
 */
#pragma once

namespace cc
{

template<typename T>
class Vector
{
private:
    unsigned int size_;
    unsigned int capacity_;
    T* buffer_;

public:
    explicit Vector(unsigned int capacity = 0)
        : size_(0)
        , capacity_(capacity)
        , buffer_(static_cast<T*>(::operator new(capacity_ * sizeof(T))))
    {
    }

    explicit Vector(std::initializer_list<T> list)
        : size_(0)
        , capacity_(list.size())
        , buffer_(static_cast<T*>(::operator new(capacity_ * sizeof(T))))
    {
        for (auto& elem : list)
        {
            emplace_back(elem);
        }
    }

    ~Vector()
    {
        for (unsigned int i = 0; i < size_; ++i)
        {
            buffer_[i].~T();
        }
        ::operator delete(buffer_);
    }

    Vector(const Vector& other)
        : size_(0)
        , capacity_(other.capacity_)
        , buffer_(static_cast<T*>(::operator new(capacity_ * sizeof(T))))
    {
        for (unsigned int i = 0; i < other.size_; ++i)
        {
            emplace_back(other.buffer_[i]);
        }
    }

    Vector& operator=(Vector& other)
    {
        if (buffer_ != other.buffer_)
        {
            Vector<T> temp(other);
            other.swap(*this);
        }
        return *this;
    }

    Vector(Vector&& other) noexcept
        : size_(0)
        , capacity_(0)
        , buffer_(nullptr)
    {
        other.swap(*this);
    }

    Vector& operator=(Vector&& other) noexcept
    {
        other.swap(*this);
        return *this;
    }

    T& operator[](unsigned int index)
    {
        return buffer_[index];
    }

    const T& operator[](unsigned int index) const
    {
        return buffer_[index];
    }

    T* data() const
    {
        return buffer_;
    }

    T* begin() const
    {
        return buffer_;
    }

    T* end() const
    {
        return buffer_ + size_;
    }

    unsigned int size() const
    {
        return size_;
    }

    bool empty() const
    {
        return size_ == 0;
    }

    void push_back(const T& value)
    {
        emplace_back(value);
    }

    template<typename ... Args>
    T& emplace_back(Args&& ... args)
    {
        if (size_ == capacity_)
        {
            capacity_ = capacity_ * 2 + 1;
            realloc();
        }

        new (buffer_ + size_) T(std::forward<Args>(args)...);
        return buffer_[size_++];
    }

    void pop_back()
    {
        buffer_[size_].~T();
        --size_;
    }

    const T& front() const
    {
        return buffer_[0];
    }

    const T& back() const
    {
        return buffer_[size_ - 1];
    }

    void clear()
    {
        for (unsigned int i = 0; i < size_; ++i)
        {
            pop_back();
        }
    }

    void resize(unsigned int count, const T& elem = T())
    {
        for (unsigned int i = size_; i < count; ++i)
        {
            emplace_back(elem);
        }

        for (unsigned int i = size_; i > count; --i)
        {
            pop_back();
        }
    }

    void swap(Vector& other) noexcept
    {
        SwapInternal(capacity_, other.capacity_);
        SwapInternal(size_, other.size_);
        SwapInternal(buffer_, other.buffer_);
    }

private:

    template<typename AnyType>
    void SwapInternal(AnyType& a, AnyType& b)
    {
        AnyType temp(std::move(a));
        a = std::move(b);
        b = std::move(temp);
    }

    void realloc()
    {
        Vector<T> expanded(capacity_);
        for (unsigned int i = 0; i < size_; ++i)
        {
            expanded.emplace_back(std::move(buffer_[i]));
        }
        expanded.swap(*this);
    }
};

}