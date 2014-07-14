//! \file thread_quickstart.cpp
//! \author Jeff Benshetler
//! \date 2012-05-18
//! Licensed under \link http://www.boost.org/LICENSE_1_0.txt Boost Software License 1.0 \endlink

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <boost/thread.hpp>
#include <boost/date_time.hpp>
#include <boost/bind.hpp>
#include "Elapsed.hpp"

const std::string RowRowRow("Row row row your boat gently down the stream");
const std::string Teapot("I'm a little teapot short and stout");

void sing(const std::string& lyrics,boost::posix_time::time_duration interval,bool indent=false) {
	std::istringstream iss;
	iss.str(lyrics);
	std::string current;
	do {
		iss >> current;
		if (iss) {
			// extra spaces make it easier to read when interleaved by threading
			if (indent)
				std::cout << "\t\t";
			std::cout  << current << "\n"; 
			boost::this_thread::sleep( interval );
		} // end if
	} while ( !iss.bad() && !iss.eof() );
	std::cout << "\n";
} // end sing()


class Singer {
	std::string m_lyrics;
	boost::posix_time::time_duration m_interval;
	bool m_indent;
public:
	Singer(const std::string& lyrics,boost::posix_time::time_duration interval,bool indent) 
		: 
		m_lyrics(lyrics),
		m_interval(interval),
		m_indent(indent)
	{

	} // end constructor
	void perform() {
		sing(m_lyrics,m_interval,m_indent);
	} // end perform()
}; // end class Singer


int main(int argc,char* argv[]) {
	using namespace boost::posix_time;

	boost::thread_group tgroup;

	time_duration interval( milliseconds(250) );
	auto delay( milliseconds(60) );

	// "sing" with a function
	//sing( RowRowRow, interval );
	tgroup.create_thread( boost::bind( &sing, RowRowRow, interval, false) );

	
	// delay
	boost::this_thread::sleep( delay );

	// "sing" with a member function
	Singer teapotSinger(Teapot,interval,true);
	//teapotSinger.perform();
	tgroup.create_thread( boost::bind( &Singer::perform, &teapotSinger ) );

	tgroup.join_all();

	return 0;
} // end main()