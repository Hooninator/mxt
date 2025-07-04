#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <chrono>

#include <cuda_runtime.h>

namespace mxt
{

class Profiler
{


public:

    Profiler()
    {
    }


    void add_stat(const char * name, double val)
    {
        std::string name_str(name);

        if (stats.contains(name_str))
        {
            stats[name_str].push_back(val);
        }
        else
        {
            stats[name_str] = {val};
        }
    }


    void stats_to_csv(const char * path)
    {
        std::ofstream ofs;
        ofs.open(path);

        std::vector<std::string> names;

        for (auto& kvp : stats)
        {
            auto name = kvp.first;
            names.push_back(name);
        }

        size_t n;
        for (auto& name : names)
        {
            ofs<<name<<",";
            n = stats[name].size();
        }
        ofs<<std::endl;

        for (size_t i = 0; i < n; i++)
        {
            for (auto& name : names)
            {
                const auto& stat_vec = stats[name];
                ofs<<stat_vec[i]<<",";
            }
            ofs<<std::endl;
        }

        stats.clear();

        ofs.close();
    }


    void start_timer(const char * name)
    {
        std::string name_str(name);

        if (active_timers.contains(name_str))
        {
            if (active_timers[name_str]->is_active())
            {
                std::cerr<<"Timer "<<name_str<<" already active."<<std::endl;
                return;
            }
            else
            {
                active_timers[name_str]->start();
            }
        }
        else
        {
            active_timers[name_str] = std::make_unique<Timer>(Timer(name_str));
            active_timers[name_str]->start();
        }
    }


    void stop_timer(const char * name)
    {
        std::string name_str(name);

        if (active_timers.contains(name_str))
        {
            active_timers[name_str]->stop();
        }
        else
        {
            std::cerr<<"Tried to stop timer "<<name<<" that is either non-existent or not running"<<std::endl;
        }
    }


    void commit_timers()
    {
        for (auto & [timer_name, timer] : active_timers)
        {
            committed_timers[timer_name].push_back(std::move(timer));
        }
        active_timers.clear();
    }


    void print_timer(const char * name)
    {
        std::string name_str(name);
        if (!active_timers.contains(name_str))
        {
            std::cerr<<"Timer "<<name<<" not in active timers."<<std::endl;
        }
        else
        {
            std::cout<<"\t["<<name<<"]: "<<active_timers[name_str]->elapsed<<"s"<<std::endl;
        }
    }


    void print_timers()
    {
        for (auto& [name, timer] : active_timers)
        {
            print_timer(name.c_str());
        }
    }


    void timers_to_csv(const char * path)
    {
        std::ofstream ofs;
        ofs.open(path);

        std::vector<std::string> names;

        for (auto& kvp : committed_timers)
        {
            auto name = kvp.first;
            names.push_back(name);
        }

        size_t n;
        for (auto& name : names)
        {
            ofs<<name<<",";
            n = committed_timers[name].size();
        }
        ofs<<std::endl;

        for (size_t i=0; i<n; i++)
        {
            for (auto& name : names)
            {
                ofs<<committed_timers[name][i]->elapsed<<",";
            }
            ofs<<std::endl;
        }

        ofs.close();

        committed_timers.clear();
    }


private:

    struct Timer
    {

        using clock_t = std::chrono::high_resolution_clock;

        Timer(std::string name):
            name(name),
            elapsed(0.0),
            active(false)
        {
        }


        inline clock_t::time_point now()
        {
            return clock_t::now();
        }


        void start()
        {
            active = true;
            start_t = now();
        }


        void stop()
        {
            end_t = now();
            double elapsed_this = (std::chrono::duration_cast<std::chrono::duration<double>>(end_t - start_t).count());
            elapsed += elapsed_this;
            active = false;
        }


        inline bool is_active()
        {
            return active;
        }


        std::string name;
        double elapsed;
        bool active;
        clock_t::time_point start_t;
        clock_t::time_point end_t;
    };


    std::map<std::string, std::unique_ptr<Timer>> active_timers;
    std::map<std::string, std::vector<std::unique_ptr<Timer>>> committed_timers;
    std::map<std::string, std::vector<double>> stats;

};



} //mxt

#endif
