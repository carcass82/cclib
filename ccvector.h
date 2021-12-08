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
    public:
        using value_type = T;
        using size_type = size_t;

        explicit Vector()
            : size_(0)
            , capacity_(16)
            , buffer_(static_cast<T*>(::operator new(capacity_ * sizeof(T))))
        {
        }

        explicit Vector(size_type count)
            : size_(count)
            , capacity_(count)
            , buffer_(new T[count])
        {
        }

        explicit Vector(size_type count, const T& elem)
            : size_(0)
            , capacity_(count)
            , buffer_(static_cast<T*>(::operator new(capacity_ * sizeof(T))))
        {
            for (size_type i = 0; i < count; ++i)
            {
                new (buffer_ + size_++) T(elem);
            }
        }

        explicit Vector(std::initializer_list<T> list)
            : size_(0)
            , capacity_(list.size())
            , buffer_(static_cast<T*>(::operator new(capacity_ * sizeof(T))))
        {
            for (auto&& it = list.begin(); it != list.end(); ++it)
            {
                new (buffer_ + size_++) T(*it);
            }
        }

        ~Vector()
        {
            for (size_type i = 0; i < size_; ++i)
            {
                buffer_[size_ - i - 1].~T();
            }

            ::operator delete(buffer_);
        }

        Vector(const Vector& other)
            : size_(0)
            , capacity_(other.capacity_)
            , buffer_(static_cast<T*>(::operator new(capacity_ * sizeof(T))))
        {
            for (size_type i = 0; i < other.size_; ++i)
            {
                new (buffer_ + size_++) T(other.buffer_[i]);
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

        T& operator[](size_type index)
        {
            return buffer_[index];
        }

        const T& operator[](size_type index) const
        {
            return buffer_[index];
        }

        T& at(size_type index)
        {
            return buffer_[index];
        }

        const T& at(size_type index) const
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

        size_type size() const
        {
            return size_;
        }

        size_type capacity() const
        {
            return capacity_;
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
                realloc(capacity_ * 2 + 1);
            }

            new (buffer_ + size_) T(std::forward<Args>(args)...);

            return buffer_[size_++];
        }

        void pop_back()
        {
            buffer_[size_--].~T();
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
            for (size_type i = 0; i < size_; ++i)
            {
                pop_back();
            }
        }

        void resize(size_type count, const T& elem = T())
        {
            for (size_type i = size_; i < count; ++i)
            {
                emplace_back(elem);
            }

            for (size_type i = size_; i > count; --i)
            {
                pop_back();
            }
        }

        void reserve(size_type capacity)
        {
            if (capacity > capacity_)
            {
                realloc(capacity);
            }
        }

        void shrink_to_fit()
        {
            realloc(size_);
        }

        void swap(Vector& other) noexcept
        {
            std::swap(capacity_, other.capacity_);
            std::swap(size_, other.size_);
            std::swap(buffer_, other.buffer_);
        }

    private:
        size_type size_;
        size_type capacity_;
        T* buffer_;

        Vector(size_type capacity, bool add_uninitialized)
            : size_(0)
            , capacity_(capacity)
            , buffer_(static_cast<T*>(::operator new(capacity_ * sizeof(T))))
        {
        }

        void realloc(size_type new_capacity)
        {
            Vector<T> expanded(new_capacity, true);
            for (size_type i = 0; i < size_; ++i)
            {
                expanded.emplace_back(std::move(buffer_[i]));
            }
            expanded.swap(*this);
        }
    };

}
