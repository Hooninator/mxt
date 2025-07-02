#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <map>
#include <vector>
#include <string>
#include <memory>

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

        for (auto& name : names)
        {
            ofs<<name<<",";
        }
        ofs<<std::endl;

        for (auto& name : names)
        {
            auto stat_vec = stats[name];
            for (double stat: stat_vec)
            {
                ofs<<stat<<",";
            }
            ofs<<std::endl;
        }


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


    inline void print_timer(const char * name)
    {
        std::cout<<"["<<name<<"]"<<active_timers[std::string(name)]<<"s"<<std::endl;
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

        for (auto& name : names)
        {
            ofs<<name<<",";
        }
        ofs<<std::endl;

        for (auto& name : names)
        {
            for (auto& timer : committed_timers[name])
            {
                ofs<<timer->elapsed<<",";
            }
            ofs<<std::endl;
        }


        ofs.close();
    }


private:

    struct Timer
    {
        Timer(std::string name):
            name(name),
            elapsed(0.0),
            active(false)
        {
            cudaEventCreate(&start_t);
            cudaEventCreate(&end_t);
        }


        void start()
        {
            cudaEventRecord(start_t);
            active = true;
        }


        void stop()
        {
            cudaEventRecord(end_t);
            float inc;
            cudaEventElapsedTime(&inc, start_t, end_t);
            elapsed += inc;
            active = false;
        }


        inline bool is_active()
        {
            return active;
        }


        ~Timer()
        {
            cudaEventDestroy(start_t);
            cudaEventDestroy(end_t);
        }


        std::string name;
        float elapsed;
        bool active;
        cudaEvent_t start_t;
        cudaEvent_t end_t;
    };


    std::map<std::string, std::unique_ptr<Timer>> active_timers;
    std::map<std::string, std::vector<std::unique_ptr<Timer>>> committed_timers;
    std::map<std::string, std::vector<double>> stats;

};



} //mxt

#endif
