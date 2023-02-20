# User Story 3 - Process/Environmental Engineer 
## Use Case
- Reduce over-purchasing of chargeable batteries for their vehicle fleet and reduce waste production/management costs.
- From the consumption perspective, from the production perspective → Warranty
Looking at the model for prac use case in an Environmental Industrial setting. 1) Manufacturer offering warranty 2) Keeping track of when/the time battery should be recycled/replaced.
## Background
Hugh wants to reduce over-purchasing of chargeable batteries for their vehicle fleet and __reduce waste production/management costs__, as well as be able to predict when the best “time” to recycle the materials within a battery. Hugh is really good at Excel!!! That’s it though, he would not find a raw script useful. Not even VBL….so I guess not that good at Excel. Hugh values a simple user interface that would have a clear and relatively short SOP so that he can train entry level employees to do it for him. A tool that generates results that can be related to cost directly would also be really useful for selling this service to a client.

"""

The specific specs dont need to be here for now. 

BG: New data has shown that exposure to heat and the use of fast charging promote the degradation of Li-ion batteries more than age and actual use, and that the average electric vehicle battery will retain 90% of its initial capacity after six years and six months of service. For example, the battery in a Nissan Leaf will degrade twice as fast as the battery in a Tesla, because the Leaf does not have an active cooling system for its battery.

Requirements: Easily searchable specs of a commercially available battery. Here is an example I found of a car battery that can be purchased (it is fast charging….) but here is the info available:
Which type of battery you would want to buy. Req should go more into the details, and 
●	Battery Chemistry : Lithium Iron Phosphate seems like the most commonly available based on a quick Goog
●	Ampere-Hour Capacity: 60 Ah , which will deliver a current of 3A for 20 hours (this would be the constant current mode for discharge? Or we could at least assume it is constant in this case, maybeeee look at the “non constant” data too)
●	Current Modes for discharge : 100A max continuous discharge, 750A max 2 second pulse, 650A max 5 sec pulse. The flat discharge voltage curve provides a 75% bigger capacity than a 60Ah SLA battery
●	Energy in Watt hours : 720 Wh
●	Battery Lifespan : Up to 80% capacity for 2,000 cycles in recommended conditions. The typical SLA has 500 cycles. Dakota Lithium batteries last so long that the price per use is a fraction of traditional batteries
●	Operating Temperature : Ideal for rugged & harsh environments. Much better than SLA or other lithium batteries. -20°F min, +150°F max optimal operating temps (battery performs well down to -20°F). Avoid charging below 32°F. Discharge cut off at - 20°F/-20°C. Charge cut off at 23°F/-5°C. BMS high temp cut off at 167°F / 75°C

"""

## Component Design - Describe what the system does?
- System: Prompts for battery voltage, Ampere-Hour Capacity, Available Modes for Discharge, approx. operating temperature or range of temperatures for the process of interest (or ideally process of interest and manual would show what range of temp is used for each process).
- System: Also has prompts for units of Temperature but certain prompts will have default units that cannot be changed
- User: Fills out prompts based on purchased batteries or batteries looking to be purchased
- System: Models capacity over time for each experiment set in the NASA dataset as the “training set”, generates data based on given parameters to create the “modeled set” shows plots for reference. Manual will include an SOP for what the plot is showing (we love visualizations)
(Practically if this was an ongoing project you could add more to the training set over time and add classifications, I was previously a data annotator for a data science team and we were scraping and generating data for them constantly to train their NL processing model for each patch or new feature)
-  User: Prompts for a report in the form of a spreadsheet to print out and show client or save for reference as they take inventory (they have to do this every year for all of their batteries anyways)

